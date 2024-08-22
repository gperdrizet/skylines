import sys
import config
import pathlib
import tensorflow as tf
import functions.data_functions as data_funcs
import functions.training_functions as training_funcs
import absl.logging

# Clean up STDOUT
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == '__main__':

    # collect command line arguments
    args=sys.argv[1:]
    resume=str(args[0])
    resume_run_date=str(args[1])

    if resume == 'True':
        path_date=resume_run_date

    elif resume == 'False':
        path_date=config.CURRENT_DATE
        
    # Check available GPUs
    print("Num GPUs Available:", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # Construct paths for run
    model_checkpoint_dir=f'{config.PATH}/data/{path_date}/training_checkpoints'
    specimen_dir=f'{config.PATH}/data/{path_date}/specimens'
    image_output_dir=f'{config.PATH}/data/{path_date}/gan_output'

    # Create or clear output directories as appropriate
    _=data_funcs.prep_output_dir(model_checkpoint_dir, resume)
    _=data_funcs.prep_output_dir(specimen_dir, resume)
    _=data_funcs.prep_output_dir(image_output_dir, resume)

    # Make TensorFlow dataset from image data
    train_ds, image_count=data_funcs.prep_data(
        config.TRAINING_IMAGE_DIR,
        config.BATCH_SIZE,
        config.IMAGE_DIM
    )

    # Get saved checkpoints, if any:
    checkpoints=list(pathlib.Path(model_checkpoint_dir).glob('generator_model_f*'))

    # Discard the last saved checkpoint because one of the models may have an incomplete save
    checkpoints=checkpoints[:-1]

    # Prepare the models for resume or fresh start
    
    # If we only have one GPU or are explicitly not using parallelism, prep the models
    # outside of a tf.distribute strategy
    if config.GPU_PARALLELISM == None: # or len(tf.config.experimental.list_physical_devices('GPU')) == 1:

        print('Running on single GPU')

        result=training_funcs.prepare_models(
                resume,
                checkpoints,
                resume_run_date,
                model_checkpoint_dir,
                image_output_dir,
                config.GENERATOR_LEARNING_RATE,
                config.LATENT_DIM,
                config.GANN_LEARNING_RATE
        )

    # If we have more than one GPU and a parallelism strategy was set in config.py, set up a scope for it
    elif config.GPU_PARALLELISM != None: # and len(tf.config.experimental.list_physical_devices('GPU')) > 1:

        print(f'Running on {len(tf.config.experimental.list_physical_devices("GPU"))} GPUs with {config.GPU_PARALLELISM} strategy')

        if config.GPU_PARALLELISM == 'mirrored':
            strategy=tf.distribute.MirroredStrategy(devices=config.GPUS)

        if config.GPU_PARALLELISM == 'central storage':
            strategy=tf.distribute.experimental.CentralStorageStrategy(compute_devices=config.GPUS)
        
        with strategy.scope():
            result=training_funcs.prepare_models(
                    resume,
                    checkpoints,
                    resume_run_date,
                    model_checkpoint_dir,
                    image_output_dir,
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
        model_checkpoint_dir,
        config.CHECKPOINT_SAVE_FREQUENCY,
        config.PROJECT_NAME,
        image_output_dir,
        config.BENCHMARK_DATA_DIR,
        config.GPU_PARALLELISM
    )