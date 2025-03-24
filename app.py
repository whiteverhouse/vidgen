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

        

if __name__=='__main__':
    app.run(debug=True)