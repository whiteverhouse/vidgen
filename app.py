from flask import Flask, render_templates, request, jsonify
import logging
from datetime import datetime
import os

app=Flask(__name__)
logging.basicConfig(
    filename='app.log', level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)

@app.route('/')
def index():
    try:
        return render_templates('index.html')
    except Exception as e:
        logging.error(f"render page gives error:{str(e)}")
        return "Error rendering index.html",500

        
@app.route('/generate_outline', methods=['POST'])
def create_outline():
    try:
  
        prompt = request.json.get('prompt')
        #if received empty input from user
        if not prompt:
            return jsonify({"error": "No prompt is received."}), 400
            
        outline = generate_outline(prompt)
        return jsonify(outline)
        
    except Exception as e: #catch error that werent handled by generate_outline()
        logging.error(f"Error generating outline: {str(e)}")
        return jsonify({"error": str(e)}), 500


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

if __name__=='__main__':
    app.run(debug=True)
