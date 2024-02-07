import os

PROJECT_NAME = 'skylines'

# get path to this config file, we will use this
# to define other paths to data files etc.
path = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = f'{path}/image_datasets'
RAW_IMAGE_DIR = f'{IMAGE_DIR}/raw_images'
PROCESSED_IMAGE_DIR = f'{IMAGE_DIR}/training_images'
TRAINING_IMAGE_DIR = PROCESSED_IMAGE_DIR
MODEL_CHECKPOINT_DIR = f'{path}/training_checkpoints'
SPECIMEN_DIR = f'{path}/specimens'
VIDEO_FRAME_LATENT_POINTS = f'{path}/gan_output/latent_points.pkl'

# Data pipeline parameters
MAX_CONCURANCY = 2
IMAGE_DIM = 1024
SHUFFLE_BUFFER = 50

# DCGANN parameters
GPUS = [
    '/job:localhost/replica:0/task:0/device:GPU:0',
    '/job:localhost/replica:0/task:0/device:GPU:1',
    '/job:localhost/replica:0/task:0/device:GPU:2',
    '/job:localhost/replica:0/task:0/device:GPU:3'
]

GPU_PARALLELISM = 'central storage'
LATENT_DIM = 100
DISCRIMINATOR_LEARNING_RATE = 0.00005
GENERATOR_LEARNING_RATE = 0.00005
GAN_LEARNING_RATE = 0.00005
BATCH_SIZE = int(3 * len(GPUS))
EPOCHS = 100000

CHECKPOINT_SAVE_FREQUENCY = 500
