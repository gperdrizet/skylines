import sys
import tensorflow as tf
import config
import functions.training_functions as training_funcs

if __name__ == '__main__':

    # collect command line arguments
    args = sys.argv[1:]
    checkpoint = int(args[0])
    num_images = int(args[1])

    # Assign job first available GPU
    with tf.device('/gpu:0'):

        # load model from checkpoint
        model_checkpoint = f'{config.MODEL_CHECKPOINT_DIR}/generator_model_f{checkpoint:07d}'
        generator_model = tf.keras.models.load_model(model_checkpoint)

        # Check its architecture
        print(generator_model.summary())

        # generate images
        for i in range(num_images):
            filename = f'{config.SPECIMEN_DIR}/{checkpoint}.{i}.jpg'
            training_funcs.save_specimen(generator_model, config.LATENT_DIM, filename)
