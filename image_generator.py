from typing import List, Dict, Optional, Union
import torch
from datetime import datetime
import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
import time
from tqdm import tqdm  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
from config_manager import ConfigManager

class ImageGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.config_manager = ConfigManager()
        self.available_models = self.config_manager.get_model_config()
        
        self.model = None
        self.current_model_key = None

    def cleanup_model(self):
        if self.model is not None:
            # move model to cpu and free up gpu memory, cut python reference to clean up
            self.model.to('cpu')
            del self.model
            self.model= None

            #tell pytorch to release unused gpu memory and force garbage collection
            torch.cuda.empty_cache()
            gc.collect()

    def load_model(self, model_key: str):
        try:
            # clean up old model then load new model so GPU memory is utilized
            if self.current_model_key !=model_key:
                self.cleanup_model()

            model_info = self.available_models.get(model_key)
            if not model_info:
                raise ValueError(f"Model {model_key} not available")
            
            if self.current_model_key == model_key and self.model is not None:
                return


            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            if model_info["pipeline"] == "StableDiffusionXLPipeline":
                from diffusers import StableDiffusionXLPipeline
                self.model = StableDiffusionXLPipeline.from_pretrained(
                    model_info["path"],
                    torch_dtype=self.torch_dtype
                )
            else:
                from diffusers import StableDiffusionPipeline
                self.model = StableDiffusionPipeline.from_pretrained(
                    model_info["path"],
                    torch_dtype=self.torch_dtype
                )
            
            self.model.to(self.device)
            self.current_model_key = model_key
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def get_supported_loras(self, model_key: str) -> List[str]:
        model_info = self.available_models.get(model_key)
        if not model_info:
            return []
        return model_info["supported_loras"]

    def generate_images(self, prompts: List[str], negative_prompts: List[str], config: Dict) -> List[str]:
        try:
            image_paths = []
            for prompt, negative_prompt in zip(prompts, negative_prompts):
                seed = int(config.get("seed")) if config.get("seed") is not None else None
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(seed)
                
                image = self.model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=config.get("width", 512),
                    height=config.get("height", 512),
                    num_inference_steps=int(config.get("num_inference_steps", 20)),
                    guidance_scale=float(config.get("guidance_scale", 7.5)),
                    generator=generator
                ).images[0]
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img_{timestamp}.png"
                path = os.path.join("static", "generated_image", filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                image.save(path)
                image_paths.append(path)
                
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            return image_paths
                
        except Exception as e:
            logging.error(f"Error generating images: {str(e)}")
            raise

    def apply_lora(self, lora_key: str = "none") -> None:
        try:
            if not self._check_lora_compatibility():
                raise ValueError("Current model doesn't support LoRA")
            
            self._cleanup_existing_lora()
            
            if lora_key == "none":
                return
            
            lora_path = self._validate_lora_path(lora_key)
            self._load_lora_weights(lora_path)

            if self._can_fuse_lora():
                self._fuse_lora_weights()
            
        except Exception as e:
            self._handle_lora_error(e)
        
    def _check_lora_compatibility(self):
        return hasattr(self.model, 'load_lora_weights')
    
    def _cleanup_existing_lora(self):
        if hasattr(self.model, 'unload_lora_weights'):
            self.model.unload_lora_weights()

    def _validate_lora_path(self,lora_key:str)->str:
        lora_configs=self.config_manager.get_lora_config()
        if not lora_configs:
            raise ValueError(f"LoRA key '{lora_key}' not found in config.json")
        
        lora_relative_path=lora_configs.get("path")
        if not lora_relative_path:
            raise ValueError(f"No path configured for LoRA key '{lora_key}' in config.json")
        
        lora_path=os.path.join("LoRA",lora_relative_path)

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at path:{lora_path}")
        return lora_path

    def _handle_lora_error(self, error: Exception):
        logging.error(f"LoRA error: {str(error)}")
        if isinstance(error, (ValueError, FileNotFoundError)):
            raise error
        raise ValueError(f"Failed to apply LoRA: {str(error)}")

    # def get_available_configs(self) -> Dict:
    #     return {
    #         "models": list(self.available_models.keys()),
    #         "parameters": {
    #             "width": {
    #                 "type": "int", "min": 256,"max": 1024, "step": 64,
    #                 "default": 512
    #             },
    #             "height": {
    #                 "type": "int","min": 256,"max": 1024,"step": 64,
    #                 "default": 512
    #             },
    #             "num_inference_steps": {
    #                 "type": "int", "min": 20, "max": 100,"step": 5,
    #                 "default": 20
    #             },
    #             "guidance_scale": {
    #                 "type": "float", "min": 1.0, "max": 20.0,"step": 0.5,
    #                 "default": 7.5
    #             },
    #             "seed":{
    #                 "type":"int","min": -1, "max": 2147483647,"step": 1,
    #                 "default":-1
    #             }
    #         }
    #     }

    def to(self, device):
        try:
            logging.info(f"Switching to device: {device}")
            self.device = device
            
            if self.model:
                self.model.to(device)                    
            return True
        except Exception as e:
            logging.error(f"Error switching device: {str(e)}")
            raise
