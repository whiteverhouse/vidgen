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

if __name__=='__main__':
    app.run(debug=True)
