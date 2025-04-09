from typing import Dict
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from pathlib import Path
import os
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import Image

class VideoGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.output_dir = Path("static/generated_video")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        logging.info("VideoGenerator initialized")
        
    def load_model(self):
        """Load the SVD model"""
        try:
            logging.info("Loading Stable Video Diffusion model")
            with tqdm(total=100, desc="Loading SVD model") as pbar:
                self.model = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=self.torch_dtype,cache_dir="model_cache"
                )
                pbar.update(75)
                
                self.model.to(self.device)
                pbar.update(25)
                
            logging.info("SVD model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading SVD model: {str(e)}")
            raise
    
    def generate_video(self, image_paths, config: Dict):
        try:
            if self.model is None:
                self.load_model()
                
            videos = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert and validate configuration parameters
            num_frames = int(config.get("num_frames", 16))
            fps = int(config.get("fps", 8))
            # video_inference_steps = int(config.get("video_inference_steps", 25))
            motion_bucket_id = int(config.get("motion_bucket", 127))
            # vid_guidance_scale = float(config.get("vid_guidance_scale", 1.0))
            vid_seed = int(config.get("vid_seed")) if config.get("vid_seed") is not None else None
            generator = torch.manual_seed(vid_seed)
            with tqdm(total=len(image_paths), desc="Generating videos") as pbar:
                for i, img_path in enumerate(image_paths):
                    img = Image.open(img_path).convert("RGB")
                    
                    video_frames = self.model(
                        img,
                        num_frames=num_frames,
                        fps=fps,
                        motion_bucket_id=motion_bucket_id,
                        generator=generator
                    ).frames[0]
                    
                    video_path = self.output_dir / f"video_{timestamp}_{i}.mp4"
                    
                    videos.append(str(export_to_video(video_frames, video_path, fps)))
                    pbar.update(1)
            
            return videos
        except Exception as e:
            logging.error(f"Error generating video: {str(e)}")
            raise
