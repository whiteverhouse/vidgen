import json
from pathlib import Path
import logging

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self):
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}
            
    def get_model_config(self):
        return self.config.get("models", {})
        
    def get_lora_config(self):
        return self.config.get("loras", {})
        
    def get_image_parameters(self):
        return self.config.get("parameters", {}).get("image", {})
        
    def get_video_parameters(self):
        return self.config.get("parameters", {}).get("video", {})
        
    def validate_image_config(self, config):
        image_params = self.get_image_parameters()
        validated = {}
        
        for key, value in config.items():
            if key in image_params:
                param_config = image_params[key]
                validated[key] = min(
                    max(float(value), param_config["min"]), 
                    param_config["max"]
                )
                
        return validated
        
    def validate_video_config(self, config):
        video_params = self.get_video_parameters()
        validated = {}
        
        for key, value in config.items():
            if key in video_params:
                param_config = video_params[key]
                validated[key] = min(
                    max(float(value), param_config["min"]), 
                    param_config["max"]
                )
                
        return validated 