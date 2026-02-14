import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, send_file, request
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained generator model
MODEL_PATH = 'generator_final.keras'
generator = None

def get_generator():
    global generator
    if generator is None:
        if os.path.exists(MODEL_PATH):
            generator = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
    return generator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    gen = get_generator()
    noise_dim = gen.input_shape[-1]
    noise = tf.random.normal([1, noise_dim])
    
    generated_image = gen(noise, training=False)
    # Post-process: [-1, 1] -> [0, 255]
    img_array = (generated_image[0, :, :, :].numpy() * 127.5 + 127.5).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    # Encode as base64 to send to frontend
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"image": img_base64}

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888)
