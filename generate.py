import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse

def generate(model_path, num_images=1, output_file='generated_anime.png'):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    generator = tf.keras.models.load_model(model_path)
    noise_dim = generator.input_shape[-1]
    
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)

    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(1, num_images, i+1)
        img = (generated_images[i, :, :, :] * 127.5 + 127.5) / 255.0
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Generated images saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="generator_final.keras", help="Path to generator model")
    parser.add_argument("--num", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--out", type=str, default="generated_anime.png", help="Output file name")
    args = parser.parse_args()
    
    generate(args.model, args.num, args.out)
