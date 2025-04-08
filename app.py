from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os
from llm_API import generate_outline
import sys
import replicate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
import json
import time

from image_generator import ImageGenerator
from video_generator import VideoGenerator
from config_manager import ConfigManager


app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for all routes
logging.basicConfig(filename='app.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add CORS headers to allow image loading
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

image_generator = ImageGenerator()
video_generator = VideoGenerator()
config_manager = ConfigManager()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index page: {str(e)}")
        return "Error rendering index page", 500

@app.route('/generate_outline', methods=['POST'])
def create_outline():
    try:
  
        prompt = request.json.get('prompt')
        #if received empty input from user.
        if not prompt:
            return jsonify({"error": "No prompt is received."}), 400
            
        outline = generate_outline(prompt)
        return jsonify(outline)
    except Exception as e: #catch error that werent handled by generate_outline()
        logging.error(f"Error generating outline: {str(e)}")
        return jsonify({"error": str(e)}), 500

# @app.route('/get_image_configs', methods=['GET'])
# # return usable models, LoRA, and limited parameters as JSON
# def get_image_configs():
#     try:
#         configs = image_generator.get_available_configs()
#         return jsonify(configs)
#     except Exception as e:
#         logging.error(f"Error getting image configs: {str(e)}")
#         return jsonify({"error": str(e)}), 500

@app.route('/get_configs')
def get_configs():
    return jsonify({
        "models": config_manager.get_model_config(),
        "lora": config_manager.get_lora_config(),
        "image_parameters": config_manager.get_image_parameters(),
        "video_parameters": config_manager.get_video_parameters()
    })

@app.route('/generate_images', methods=['POST'])
def create_images():
    try:
        data = request.json
        # ensure passed config is within the range(e.g. width min value is 256, lower than 256 would be pluck to 256)
        config = config_manager.validate_image_config(data.get("config", {}))
        prompts=data.get('prompts')
        negative_prompts=data.get('negativePrompts')
        
        model_key = data.get('model', 'base') #get say sd15
        lora_key = data.get('lora', 'none')
        
        logging.info(f"Received image generation request with prompts: {prompts}")
        logging.info(f"Negative prompts: {negative_prompts}")
        
        if not prompts:
            return jsonify({"error": "No image prompts received"}), 400
            
        image_generator.load_model(model_key)
        image_generator.apply_lora(lora_key)
        
        image_paths = image_generator.generate_images(
            prompts=prompts,
            negative_prompts=negative_prompts if negative_prompts else None,
            config=config
        )
        image_urls = [f"/static/generated_image/{Path(path).name}" for path in image_paths]
        
        return jsonify({
            "image_paths": image_urls,
            "config": config
        })
        
    except Exception as e:
        logging.error(f"Error generating images: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/regenerate_image', methods=['POST'])
def regenerate_image():
    try:
        # default value to write less if-else and function operate as expected
        data = request.json
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt')
        config = data.get('config', {})
        image_index = data.get('image_index', 0) 
        model_key = data.get('model', 'base')
        lora_key = data.get('lora', 'none')
        
        if not prompt:
            return jsonify({"error": "No prompt received"}), 400

        image_generator.load_model(model_key)
        image_generator.apply_lora(lora_key)

        image_paths = image_generator.generate_images(
            prompts=[prompt],
            negative_prompts=[negative_prompt] if negative_prompt else None,
            config=config
        )
        image_url = f"/static/generated_image/{Path(image_paths[0]).name}"
        
        return jsonify({
            "image_path": image_url,
            "image_index": image_index
        })
        
    except Exception as e:
        logging.error(f"Error regenerating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_video', methods=['POST'])
def create_video():
    try:
        data = request.json
        config = config_manager.validate_video_config(data.get("config", {}))
        image_paths = data.get("image_paths")
        video_index = data.get("video_index", None)

        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if not image_paths:
            return jsonify({"error": "No image paths provided"}), 400
            
        torch.cuda.empty_cache()
        
        video_paths = video_generator.generate_video(
            image_paths=image_paths,
            config=config
        )
        
        response_data = {
            "video_paths": [f"/static/generated_video/{Path(path).name}" for path in video_paths]
        }
        
        if video_index is not None:
            response_data["video_index"] = video_index
            
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error generating video: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_images_replicate', methods=['POST'])
def create_images_replicate():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        
        prompts = data.get("prompts")
        negative_prompts = data.get('negativePrompts', [])

        config = config_manager.validate_image_config(data.get("config", {}))
        
        image_urls = []
        for i, prompt in enumerate(prompts):
            neg_prompt = negative_prompts[i] if i < len(negative_prompts) else ""
            output = replicate.run(
                "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                input={
                    "prompt": prompt,
                    "negative_prompt": neg_prompt,
                    "width":config.get("width", 768),
                    "height":config.get("height", 768),
                    "num_inference_steps":config.get("num_inference_steps", 25),
                    "guidance_scale":config.get("guidance_scale", 7.5),
                    "seed":config.get("seed"),
                    "apply_watermark":True, #to add watermark so it would be detected as AI0-generated work
                    "disable_safety_checker":True
                }
            )
            logging.info(f"Replicate output: {output}")
            image_url = save_single_uri_image(output, prefix=f"img_{i}")
            image_urls.append(image_url)
 
        return jsonify({
            "image_paths": image_urls
        })
        
    except Exception as e:
        logging.error(f"Error in Replicate API: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def save_single_uri_image(output, prefix="img"):
    if isinstance(output, list):
        uri = output[0]
    else:
        uri = output

    save_dir = "static/generated_image"
    os.makedirs(save_dir, exist_ok=True)
    import requests
    response = requests.get(uri)
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{prefix}_{timestamp}.png"
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "wb") as file:
            file.write(response.content)

        logging.info(f"Image successfully downloaded and saved to: {file_path}")
        return f"/static/generated_image/{file_name}"
    else:
        error_msg = f"Failed to download image: {uri}, status code: {response.status_code}"
        logging.error(error_msg)
        raise Exception(error_msg)

@app.route('/generate_video_replicate', methods=['POST'])
def create_video_replicate():
    try:
        data = request.json
        image_paths = data.get("image_paths") #https://e85b-223-19-79-74.ngrok-free.app/static/generated_image/img_0_20250330_155009.svg
        videoPrompts = data.get("videoPrompts")
        # config = config_manager.validate_video_config(data.get("config", {}))

        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if not image_paths:
            return jsonify({"error": "No image paths provided"}), 400
        
        # Kling model 1.6 std is used 
        video_urls = []
        for i, path in enumerate(image_paths):
            output = replicate.run(
                "kwaivgi/kling-v1.6-pro",
                input={
                    "prompt": videoPrompts[i],
                    "start_image": path
                    
                }
            )
            video_url = save_single_uri_vid(output,prefix=f"vid_{i}")
            video_urls.append(video_url)
        
        return jsonify({
            "video_paths": video_urls,
        })
        
    except Exception as e:
        logging.error(f"Error in Replicate video generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

# def save_vid_replicate_output(output, prefix="img"):
#     save_paths = []
#     save_dir = "static/generated_video"
#     os.makedirs(save_dir, exist_ok=True)

#     for i, item in enumerate(output):
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         file_name = f"{prefix}_{timestamp}_{i}.mp4"
#         file_path = os.path.join(save_dir, file_name)
#         with open(file_path, "wb") as file:
#             file.write(item)
#         save_paths.extend(file_path)
    
#     return save_paths

def save_single_uri_vid(uri, prefix="img"):
    save_dir = "static/generated_video"
    os.makedirs(save_dir, exist_ok=True)
    import requests
    response = requests.get(uri)
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{prefix}_{timestamp}.mp4"
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "wb") as file:
            file.write(response.content)

        logging.info(f"Video successfully downloaded and saved to: {file_path}")
        return f"/static/generated_video/{file_name}"
    else:
        error_msg = f"Failed to download video: {uri}, status code: {response.status_code}"
        logging.error(error_msg)
        raise Exception(error_msg)


if __name__ == '__main__':
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        app.run(host='0.0.0.0', port=5000)
    else:
        # Local development
        app.run(debug=True)
