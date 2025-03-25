from typing import List, Dict, Optional, Union
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import time
from tqdm import tqdm  # For progress indication
import sys
sys.path.append(r'C:\Users\11151\Downloads\fyp2024\VideoGen')

class ImageGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.output_dir = Path("static/generated_image")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track API calls
        self.last_api_call = 0
        self.min_api_interval = 1  # Minimum seconds between API calls

        self.default_config = {
            "width": 512,
            "height": 512,
            "num_inference_steps": 20, 
            "guidance_scale": 7.5,
            "seed": 42
        }
        
        # Available models from Hugging Face
        self.available_models = {
            "SD 1.4 base model": "CompVis/stable-diffusion-v1-4",
            "SD 1.5 base model": "runwayml/stable-diffusion-v1-5",
            "cyberpunk":"genai-archive/anything-v5",
            "ancientCN":"xiaolxl/GuoFeng3",
            "SD 2.1 base model": "stabilityai/stable-diffusion-2-1"
        }
        
        self.available_loras = {
            "none": None,            
            "ancientCN": "LoRA\tarot card 512x1024.safetensors",
            "cyberpunk": "LoRA\MoXinV1.safetensors"
        }
        
        self.model = None
        self.current_model_key = None
        
        logging.info(f"Initializing ImageGenerator with device: {device}")
        logging.info("ImageGenerator initialized")

        if self.device == "cuda":
            # Enable memory efficient attention
            self.default_memory_settings = {
                "attention_slice_size": 1,
                "use_tf32": True,
            }

    def _rate_limit_check(self):
        """Simple rate limiting"""
        current_time = time.time()
        if current_time - self.last_api_call < self.min_api_interval:
            wait_time = self.min_api_interval - (current_time - self.last_api_call)
            time.sleep(wait_time)
        self.last_api_call = current_time

    def load_model(self, model_key: str = "base") -> None:
        """Load model from Hugging Face Hub"""
        try:
            if self.current_model_key == model_key and self.model is not None:
                return
            
            model_id = self.available_models.get(model_key)
            if not model_id:
                raise ValueError(f"Unknown model: {model_key}")
            
            logging.info(f"Loading model: {model_id}")
            
            # Show progress during model loading
            with tqdm(total=100, desc="Loading model") as pbar:
                self.model = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    cache_dir="model_cache"  # Cache models locally
                )
                pbar.update(50)
                
                self.model.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.model.scheduler.config
                )
                pbar.update(25)
                
                if self.device == "cuda":
                    self.model.enable_attention_slicing(self.default_memory_settings["attention_slice_size"])
                    torch.backends.cuda.matmul.allow_tf32 = self.default_memory_settings["use_tf32"]
                
                self.model.to(self.device)
                pbar.update(25)
            
            self.current_model_key = model_key
            logging.info(f"Model {model_key} loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def apply_lora(self, lora_key: str = "none") -> None:
        """Apply LoRA if available"""
        try:
            if lora_key == "none":
                return
                
            lora_path = self.available_loras.get(lora_key)
            if not lora_path:
                raise ValueError(f"Unknown LoRA: {lora_key}")
            
            logging.info(f"Applying LoRA: {lora_key}")

            if os.path.exists(lora_path):
                self.model.load_lora_weights(lora_path)
                if hasattr(self.model, 'fuse_lora'):
                    self.model.fuse_lora()
                logging.info("LoRA applied and fused successfully")
            else:
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
            
        except Exception as e:
            logging.error(f"Error applying LoRA: {str(e)}")
            raise

    def generate_images(
        self,
        prompts: List[str],
        negative_prompts: List[str] = None,
        config: Optional[Dict] = None
    ) -> List[str]:
        """Generate images with progress indication"""
        try:
            logging.info(f"Starting image generation with device: {self.device}")
            logging.info(f"Prompts: {prompts}")
            logging.info(f"Config: {config}")
            
            if self.model is None:
                self.load_model()
            
            generation_config = self.default_config.copy()
            if config:
                generation_config.update(config)
            
            image_paths = []
            
            # Progress bar for all images
            with tqdm(total=len(prompts), desc="Generating images") as pbar:
                for i, prompt in enumerate(prompts):
                    logging.info(f"Starting generation for prompt {i+1}: {prompt}")
                    try:
                        self._rate_limit_check()
                        
                        if generation_config["seed"] is not None:
                            torch.manual_seed(generation_config["seed"] + i)
                        
                        negative_prompt = negative_prompts[i] if negative_prompts else None
                        
                        logging.info(f"Generating image {i+1}/{len(prompts)} with prompt: {prompt}")
                        
                        try:
                            image = self.model(
                                prompt,
                                negative_prompt=negative_prompt,
                                width=generation_config["width"],
                                height=generation_config["height"],
                                num_inference_steps=generation_config["num_inference_steps"],
                                guidance_scale=generation_config["guidance_scale"]
                            ).images[0]
                        except Exception as e:
                            logging.error(f"Error during model inference: {str(e)}")
                            raise
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        image_path = self.output_dir / f"image_{timestamp}_{i}.png"
                        
                        try:
                            image.save(image_path)
                        except Exception as e:
                            logging.error(f"Error saving image: {str(e)}")
                            raise
                            
                        image_paths.append(str(image_path))
                        pbar.update(1)
                        logging.info(f"Generated image saved to: {image_path}")
                        logging.info(f"Completed generation for prompt {i+1}")
                        
                    except Exception as e:
                        logging.error(f"Error generating image {i+1}: {str(e)}")
                        raise
            
            return image_paths
            
        except Exception as e:
            logging.error(f"Error in generate_images: {str(e)}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_available_configs(self) -> Dict:

        return {
            "models": list(self.available_models.keys()),
            "loras": list(self.available_loras.keys()),
            "parameters": {
                "width": {
                    "type": "int",
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "default": self.default_config["width"]
                },
                "height": {
                    "type": "int",
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "default": self.default_config["height"]
                },
                "num_inference_steps": {
                    "type": "int",
                    "min": 20,
                    "max": 100,
                    "step": 1,
                    "default": self.default_config["num_inference_steps"]
                },
                "guidance_scale": {
                    "type": "float",
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "default": self.default_config["guidance_scale"]
                },
                "seed": {
                    "type": "int",
                    "min": -1,
                    "max": 2147483647,
                    "default": self.default_config["seed"]
                }
            }
        }

    def to(self, device):
        self.device = device
        if self.model:
            self.model.to(device) 
