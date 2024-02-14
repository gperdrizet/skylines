import config
import pickle
import pathlib
import tensorflow as tf
import functions.data_functions as data_funcs
import functions.model_definitions as models
import functions.training_functions as training_funcs
import absl.logging

# Clean up STDOUT
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == '__main__':

    # Create or clear output directories as appropriate
    _=data_funcs.prep_output_dir(config.MODEL_CHECKPOINT_DIR, config.RESUME)
    _=data_funcs.prep_output_dir(config.SPECIMEN_DIR, config.RESUME)
    _=data_funcs.prep_output_dir(config.IMAGE_OUTPUT_DIR, config.RESUME)

    # Make TensorFlow dataset from image data
    train_ds, image_count=data_funcs.prep_data(
        config.TRAINING_IMAGE_DIR,
        config.BATCH_SIZE,
        config.IMAGE_DIM
    )

    # Get saved checkpoints, if any:
    checkpoints=list(pathlib.Path(config.MODEL_CHECKPOINT_DIR).glob('generator_model_f*'))

    # Check available GPUs
    print("Num GPUs Available:", len(
        tf.config.experimental.list_physical_devices('GPU')))
    
    # If we only have one GPU or are explicitly not using parallelism, prep the models
    # outside of a tf.distribute strategy
    if config.GPU_PARALLELISM == None or len(tf.config.experimental.list_physical_devices('GPU')) == 1:

        print('Running on single GPU')

        result=training_funcs.prepare_models(
                config.RESUME,
                checkpoints,
                config.RESUME_RUN_DATE,
                config.MODEL_CHECKPOINT_DIR,
                config.IMAGE_OUTPUT_DIR,
                config.GENERATOR_LEARNING_RATE,
                config.LATENT_DIM,
                config.GANN_LEARNING_RATE
        )

    # If we have more than one GPU and a parallelism strategy was set in config.py, set up a scope for it
    elif config.GPU_PARALLELISM != None or len(tf.config.experimental.list_physical_devices('GPU')) > 1:

        print(f'Running on {len(tf.config.experimental.list_physical_devices("GPU"))} GPUs with {config.GPU_PARALLELISM} strategy')

        if config.GPU_PARALLELISM == 'mirrored':
            strategy=tf.distribute.MirroredStrategy(devices=config.GPUS)

        if config.GPU_PARALLELISM == 'central storage':
            strategy=tf.distribute.experimental.CentralStorageStrategy(compute_devices=config.GPUS)
        
        with strategy.scope():
            result=training_funcs.prepare_models(
                    config.RESUME,
                    checkpoints,
                    config.RESUME_RUN_DATE,
                    config.MODEL_CHECKPOINT_DIR,
                    config.IMAGE_OUTPUT_DIR,
                    config.GENERATOR_LEARNING_RATE,
                    config.LATENT_DIM,
                    config.GANN_LEARNING_RATE
            )

    # Unpack result from model preparation
    latent_points=result[0]
    frame=result[1]
    discriminator_model=result[2]
    generator_model=result[3]
    gann_model=result[4]

    # train model
    training_funcs.train(
        frame,
        generator_model,
        discriminator_model,
        gann_model,
        train_ds,
        config.LATENT_DIM,
        latent_points,
        image_count,
        config.EPOCHS,
        config.BATCH_SIZE,
        config.MODEL_CHECKPOINT_DIR,
        config.CHECKPOINT_SAVE_FREQUENCY,
        config.PROJECT_NAME,
        config.IMAGE_OUTPUT_DIR
    )