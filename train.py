import tensorflow as tf
import os
import time
from model import AnimeGAN, load_dataset, generate_and_save_images

# Configuration
DATA_DIR = 'c:/Users/student/Desktop/animi_gan/animi_gan/ganyu' 
EPOCHS = 100  
BATCH_SIZE = 16  # Reduced for faster iteration speed
NOISE_DIM = 100
IMG_SIZE = 128  
NUM_EXAMPLES_TO_GENERATE = 16

def train():
    # Load data
    print("Loading dataset...")
    train_dataset = load_dataset(DATA_DIR, IMG_SIZE, IMG_SIZE, BATCH_SIZE)

    # Initialize GAN
    gan = AnimeGAN(IMG_SIZE, IMG_SIZE, NOISE_DIM)

    # Seed for consistent visualization
    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

    print("Starting training...")
    for epoch in range(EPOCHS):
        start = time.time()
        
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()

        for image_batch in train_dataset:
            gen_loss, disc_loss = gan.train_step(image_batch)
            gen_loss_avg.update_state(gen_loss)
            disc_loss_avg.update_state(disc_loss)

        # Produce images for the GIF as we go
        generate_and_save_images(gan.generator, epoch + 1, seed)

        if (epoch + 1) % 10 == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            gan.generator.save(f'checkpoints/generator_epoch_{epoch+1}.keras')

        print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec. Gen Loss: {gen_loss_avg.result():.4f}, Disc Loss: {disc_loss_avg.result():.4f}')

    # Final save
    gan.generator.save('generator_final.keras')
    print("Training complete!")

if __name__ == "__main__":
    train()
