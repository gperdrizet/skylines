import sys
import os.path
# import re
import pickle
import pathlib
# from pathlib import Path
from os import listdir
from os.path import isfile, join
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import config as conf
import functions.training_functions as training_funcs
#import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    # collect command line arguments
    args = sys.argv[1:]
    run_date = str(args[0])
    laten_point_file = str(args[1])

    # Load latent point
    with open(f'{conf.path}/data/specimens/{run_date}/{laten_point_file}', 'rb') as f:
       latent_point = pickle.load(f)

    f.close()

    print(f'Loaded latent point has shape {latent_point.shape}')

    # Prep output directory for frames
    output_path = f'{conf.path}/data/specimens/{run_date}/training_sequence'

    # Check if the directory exists
    if pathlib.Path(output_path).is_dir():

        # It already exists, so empty it
        for filename in os.listdir(output_path):
            file_path = os.path.join(output_path, filename)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

    else:

        # It doesn't exist, so create it
        pathlib.Path(output_path).mkdir()
    

    # Assign to CPU incase another training run is going in the background
    with tf.device('/CPU:0'):

        frame = 0

        model_checkpoint_dir = f'{conf.path}/data/training_checkpoints/{run_date}/'
        models = [f for f in listdir(model_checkpoint_dir) if isfile(join(model_checkpoint_dir, f))]

        for model in sorted(models):

            print(f'Generating frame {frame} from {str(model)}')

            # load model from checkpoint
            generator_model = tf.keras.models.load_model(f'{model_checkpoint_dir}/{model}')

            # generate image
            filename = f'{output_path}/{frame}'
            training_funcs.save_specimen_from_latent_point(generator_model, latent_point, filename)

            # Step frame number
            frame+=1