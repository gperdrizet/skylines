import config
# import sys
# import os
import pickle
import pathlib
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import functions.data_functions as data_funcs
import functions.model_definitions as models
import functions.training_functions as training_funcs
# from tensorflow.python.client import device_lib
# from keras.models import load_model
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == '__main__':

    # Create or clear output directories as appropriate
    _=data_funcs.prep_output_dir(config.MODEL_CHECKPOINT_DIR, config.RESUME)
    _=data_funcs.prep_output_dir(config.SPECIMEN_DIR, config.RESUME)
    _=data_funcs.prep_output_dir(config.IMAGE_OUTPUT_DIR, config.RESUME)

    print("Num GPUs Available:", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # Pick distributed training strategy, if any
    if config.GPU_PARALLELISM == 'mirrored':
        strategy=tf.distribute.MirroredStrategy(devices=config.GPUS)

    elif config.GPU_PARALLELISM == 'central storage':
        strategy=tf.distribute.experimental.CentralStorageStrategy(
            compute_devices=config.GPUS)

    else:
        strategy=None

    with strategy.scope():

        # Size of the latent space
        latent_dim=config.LATENT_DIM

        # Make TensorFlow dataset from image data
        train_ds, image_count = data_funcs.prep_data(
            config.TRAINING_IMAGE_DIR,
            config.BATCH_SIZE,
            config.IMAGE_DIM
        )

        # Get saved checkpoints, if any:
        checkpoints = list(pathlib.Path(config.MODEL_CHECKPOINT_DIR).glob('generator_model_f*'))

        # Resume prior run if appropriate
        if config.RESUME == True and len(checkpoints) > 0:

            print(f'Resuming run from {config.RESUME_RUN_DATE}')

            # Get last checkpoint file
            last_checkpoint = str(checkpoints[-1])

            # Parse step number from filename
            last_checkpoint=int(last_checkpoint.split('generator_model_f')[-1])
            print(f'Last checkpoint step: {last_checkpoint}')

            # Load up the discriminator and generator
            discriminator_model=tf.keras.models.load_model(
                f'{config.MODEL_CHECKPOINT_DIR}/discriminator_model_f{last_checkpoint:07d}')
            generator_model=tf.keras.models.load_model(
                f'{config.MODEL_CHECKPOINT_DIR}/generator_model_f{last_checkpoint:07d}')

            # Read the latent points from the last run
            with open(f'{config.IMAGE_OUTPUT_DIR}/latent_points.pkl', 'rb') as f:
                latent_points=pickle.load(f)

            # Resume the frame number
            frame=last_checkpoint

        # If we are explicitly not resuming or we don't have any checkpoints to resume
        # from, build the models fresh
        elif config.RESUME == False or len(checkpoints) == 0:

            print(f'Starting new run')

            # Create the models
            discriminator_model=models.define_discriminator(config.DISCRIMINATOR_LEARNING_RATE)
            generator_model=models.define_generator(latent_dim)

            # Make static set of points in latent space to generate
            # sample images from. Save these for later incase we need
            # to stop and restart training.
            latent_points=training_funcs.generate_latent_points(latent_dim, 9)

            with open(f'{config.IMAGE_OUTPUT_DIR}/latent_points.pkl', 'wb') as f:
                pickle.dump(latent_points, f)

            # Set the starting frame number
            frame=1

        # Build the GANN
        gann_model=models.define_gan(
            generator_model,
            discriminator_model,
            config.GAN_LEARNING_RATE
        )

        # train model
        training_funcs.train(
            frame,
            generator_model,
            discriminator_model,
            gann_model,
            train_ds,
            latent_dim,
            latent_points,
            image_count,
            config.EPOCHS,
            config.BATCH_SIZE,
            config.MODEL_CHECKPOINT_DIR,
            config.CHECKPOINT_SAVE_FREQUENCY,
            config.PROJECT_NAME,
            config.IMAGE_OUTPUT_DIR
        )