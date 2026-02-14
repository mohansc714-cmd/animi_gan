
import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, send_file, request
from PIL import Image
import io
import base64
from tensorflow.keras import layers

app = Flask(__name__)

# Re-defining the 64x64 architecture for compatibility with the existing model
def build_legacy_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

MODEL_PATH = 'generator_final.keras'
generator = None

def get_generator():
    global generator
    if generator is None:
        if os.path.exists(MODEL_PATH):
            # We build the architecture and load weights to ensure compatibility
            generator = build_legacy_generator()
            generator.load_weights(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
    return generator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    gen = get_generator()
    noise = tf.random.normal([1, 100])
    generated_image = gen(noise, training=False)
    img_array = (generated_image[0, :, :, :].numpy() * 127.5 + 127.5).astype(np.uint8)
    img = Image.fromarray(img_array).resize((384, 384), Image.NEAREST) # Scaled for visibility
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"image": img_base64}

if __name__ == '__main__':
    print("Starting Anime GAN on port 8888...")
    app.run(debug=False, host='0.0.0.0', port=8888)
