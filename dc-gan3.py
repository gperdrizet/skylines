import config
import sys
import os
import pickle
import tensorflow as tf
import functions.data_functions as data_funcs
import functions.model_definitions as models
import functions.training_functions as training_funcs
from tensorflow.python.client import device_lib

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    if config.GPU_PARALLELISM == 'mirrored':
        strategy = tf.distribute.MirroredStrategy(devices=config.GPUS)

    elif config.GPU_PARALLELISM == 'central storage':
        strategy = tf.distribute.experimental.CentralStorageStrategy(
            compute_devices=config.GPUS)

    else:
        strategy = None

    with strategy.scope():
        # size of the latent space
        latent_dim = config.LATENT_DIM

        train_ds, image_count = data_funcs.prep_data(
            config.TRAINING_IMAGE_DIR,
            config.BATCH_SIZE,
            config.IMAGE_DIM
        )

        # create the models
        discriminator_model = models.define_discriminator(
            config.DISCRIMINATOR_LEARNING_RATE
        )

        print(f'\nDiscriminator model:\n{discriminator_model.summary()}')

        generator_model = models.define_generator(
            latent_dim,
            config.GENERATOR_LEARNING_RATE
        )

        print(f'\nGenerator model:\n{generator_model.summary()}')

        gan_model = models.define_gan(
            generator_model,
            discriminator_model,
            config.GAN_LEARNING_RATE
        )

        # generate static set of points in latent space to generate
        # training video frames. Save these for later incase we need
        # to stop and restart training
        latent_points = training_funcs.generate_latent_points(latent_dim, 9)

        with open(config.VIDEO_FRAME_LATENT_POINTS, 'wb') as f:
            pickle.dump(latent_points, f)

        f.close()

        # train model
        frame = 1
        training_funcs.train(
            frame,
            generator_model,
            discriminator_model,
            gan_model,
            train_ds,
            latent_dim,
            latent_points,
            image_count,
            config.EPOCHS,
            config.BATCH_SIZE,
            config.CHECKPOINT_SAVE_FREQUENCY,
            config.PROJECT_NAME
        )