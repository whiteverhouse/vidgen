from typing import Dict, Any, List
import json
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# read env variable from .env
load_dotenv() 

logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#make sure api key is set up in .env
api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise ValueError("MOONSHOT_API_KEY not found in .env file")

try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )
    logging.info("OpenAI client initialized successfully")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {str(e)}")
    raise


def generate_outline(prompt: str) -> Dict[str, Any]:
    """
    Args:
        prompt (str): user's input
    
    Returns:
        Dict[str, Any]: JSON formatted video outline
    """
    try:
        logging.info(f"Sending request with prompt: {prompt[:50]}...")
        
        # rate limiting by forcing a 1 sec delay between requests
        time.sleep(1)          
        system_prompt = """
        You are a professional English video story board planner. Generate scenes for the video in English. You must follow below JSON format:
        {
            "outline": {
                "title": "Video Title",
                "scenes": [
                    {
                        "scene_number": 1,
                        "description": "Scene description",
                        "image_prompt": str,
                        "negative_prompt":str,
                        “video_prompt":str,
                        "narration":str
                    }
                ]
            }
        }
        
        Make sure each scene must have scene_number, description,simple and engaging narration, good image_prompt and negative_prompt for image generation,and “video_prompt for video generation prompt. 
        The prompts should be detailed, high-quality, creative, and aesthetic image generation prompt that tells a story with different camera angles and perspectives. 
        The negative_prompt is to rule out deformed body, NSFW or inappropriate content, low quality, blurry, unclear images, watermarks.
        Each prompt should only be made of 77 tokens to accomadate the input limit of stable diffusion pipeline.
        """
        
        completion = client.chat.completions.create(
            model="moonshot-v1-8k", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # see if a JSON object is returned
        content = json.loads(completion.choices[0].message.content)
        logging.info("Successfully generated outline.")
        logging.info(f"Generated content: {json.dumps(content, indent=2)}")
        
        return content
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {str(e)}")
        logging.error(f"Raw response:{completion.choices[0].message.content}")
        return {"error":"Invalid JSON response from API"}
    except Exception as e:
        logging.error(f"Error generating outline: {str(e)}")
        if e.response.status_code ==429:
            logging.error("Rate limit reached. Waiting for retry after 2 seconds..")
            return {"error": "Rate limit reached. Please try again later."}        
        else:
            return {"error": "An unexpected error occurred."}

"""please generate a video for the chinese poetry:
静夜思·李白
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
Translation:Night Thoughts·Li Bai
The bright moon shines before my bed:
I wonder if it’s frost on the ground spread.
At the bright moon I look up,
And yearn for my old home as I lower my head."""
