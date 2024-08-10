# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import CLIPImageProcessor,T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors"
FEATURE_EXTRACTOR = "/src/feature-extractor"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        
        print("Loading Flux txt2img Pipeline")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
       
        bfl_repo = "black-forest-labs/FLUX.1-dev"
        dtype = torch.bfloat16

        transformer = FluxTransformer2DModel.from_single_file(MODEL_CACHE, torch_dtype=dtype)
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2
        self.txt2img_pipe = pipe

        # Save some VRAM by offloading the model to CPU
        vram = int(torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024))
        if vram < 50:
            print("GPU VRAM < 50Gb - Offloading model to CPU")
            self.txt2img_pipe.enable_model_cpu_offload()
        
        print("setup took: ", time.time() - start)


    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),"16:9": (1344, 768),"21:9": (1536, 640),
            "3:2": (1216, 832),"2:3": (832, 1216),"4:5": (896, 1088),
            "5:4": (1088, 896),"9:16": (768, 1344),"9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1"),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)

        guidance_scale=0.0
        num_inference_steps=4
        max_sequence_length=256

        flux_kwargs = {}
        print(f"Prompt: {prompt}")
        print("txt2img mode")
        flux_kwargs["width"] = width
        flux_kwargs["height"] = height
        pipe = self.txt2img_pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil"
        }

        output = pipe(**common_args, **flux_kwargs)

    

        output_paths = []
        for i, image in enumerate(output.images):

            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception("No Image generated, sTry running it again, or try a different prompt.")

        return output_paths
    