import pickle
import time
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy.random import randn
import matplotlib.pyplot as plt
import functions.model_definitions as models
import functions.training_functions as training_funcs

np.set_printoptions(threshold=np.inf)


# Handles resuming or not and getting models ready accordingly

def prepare_models(
    resume,
    checkpoints,
    resume_run_date,
    model_checkpoint_dir,
    image_output_dir,
    discriminator_learning_rate,
    latent_dim,
    gann_learning_rate
):
    # Resume prior run if appropriate
    if resume == 'True' and len(checkpoints) > 0:

        print(f'Resuming run from {resume_run_date}')

        # Get last checkpoint file
        last_checkpoint=str(checkpoints[-1])

        # Parse step number from filename
        last_checkpoint=int(last_checkpoint.split('generator_model_f')[-1])
        print(f'Last checkpoint step: {last_checkpoint}\n')

        # Load up the discriminator and generator
        discriminator_model=tf.keras.models.load_model(
            f'{model_checkpoint_dir}/discriminator_model_f{last_checkpoint:07d}')
        generator_model=tf.keras.models.load_model(
            f'{model_checkpoint_dir}/generator_model_f{last_checkpoint:07d}')

        # Read the latent points from the last run
        with open(f'{image_output_dir}/latent_points.pkl', 'rb') as f:
            latent_points=pickle.load(f)

        # Resume the frame number
        frame=last_checkpoint

    # If we are explicitly not resuming or we don't have any checkpoints to resume
    # from, build the models fresh
    elif resume == 'False' or len(checkpoints) == 0:

        print(f'Starting new run\n')

        # Create the models
        discriminator_model=models.define_discriminator(discriminator_learning_rate)
        generator_model=models.define_generator(latent_dim)

        # Make static set of points in latent space to generate
        # sample images from. Save these for later incase we need
        # to stop and restart training.
        latent_points=training_funcs.generate_latent_points(latent_dim, 9)

        with open(f'{image_output_dir}/latent_points.pkl', 'wb') as f:
            pickle.dump(latent_points, f)

        # Set the starting frame number
        frame=1

    # Build the GANN
    gann_model=models.define_gan(
        generator_model,
        discriminator_model,
        gann_learning_rate
    )

    return [latent_points, frame, discriminator_model, generator_model, gann_model]


# train the generator and discriminator

def train(
    frame, 
    g_model, 
    d_model, 
    gan_model, 
    dataset, 
    latent_dim, 
    latent_points, 
    image_count, 
    n_epochs, 
    n_batch,
    model_checkpoint_dir,
    checkpoint_save_frequency, 
    project_name,
    image_output_dir,
    benchmarking_data_dir,
    gpu_parallelism
):

    # Calculate how many batches we need for each epoch
    bat_per_epo=int((image_count) / n_batch)

    # Construct filename for benchmarking results
    num_gpus=len(tf.config.experimental.list_physical_devices("GPU"))
    benchmark_datafile=f'{benchmarking_data_dir}/training_benchmark_data.csv'

    # Make empty dict for benchmarking data
    benchmark_data={
        'GPU parallelism': [],
        'GPUs': [],
        'Batch size': [],
        'Training time (sec.)': []
    }

    # loop on epochs
    for i in range(n_epochs):

        iterator=iter(dataset)
        training_times=[]

        # loop on batches
        for j in range(bat_per_epo):

            # get randomly selected 'real' samples
            X_real=iterator.get_next()
            y_real=ones((n_batch, 1))

            # Start training timer
            training_start=time.time()

            # train discriminator on real samples
            d_loss1, _=d_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake=generate_fake_samples(g_model, latent_dim, n_batch)
            X_fake=X_fake

            # train discriminator on fake samples from generator
            d_loss2, _=d_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator
            X_gan=generate_latent_points(latent_dim, n_batch)

            # create inverted labels for the fake samples
            y_gan=ones((n_batch, 1))

            # train the generator via the discriminator's error
            g_loss=gan_model.train_on_batch(X_gan, y_gan)

            # Add batch time to list
            training_times.append(time.time() - training_start)

            # Save sample output images
            _=save_frame(g_model, latent_points, frame, image_output_dir)

            # summarize loss on this batch
            print(f'{project_name}-{frame}: d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}\n')

            # Increment frame number
            frame+=1

            # Save model checkpoint if needed
            if frame % checkpoint_save_frequency == 0:
                g_model.save(f'{model_checkpoint_dir}/generator_model_f{frame:07d}')
                d_model.save(f'{model_checkpoint_dir}/discriminator_model_f{frame:07d}')

            # Save benchmarking data every 100 batches
            if frame % 2 == 100:

                # Make dataframe from benchmarking data
                benchmark_data['Batch time (sec.)']=training_times
                benchmark_data['GPU parallelism']=[str(gpu_parallelism)] * len(training_times)
                benchmark_data['GPUs']=[str(num_gpus)] * len(training_times)
                benchmark_data['Batch size']=[str(n_batch)] * len(training_times)

                training_times=[]

                new_data_df=pd.DataFrame(benchmark_data)

                # Check to see if we already have data for this config
                if os.path.isfile(benchmark_datafile) == True:

                    # The datafile already exists read old data and append
                    old_data_df=pd.read_csv(benchmark_datafile)
                    output_df=pd.concat([old_data_df, new_data_df], axis=0)

                else:
                    # If we don't have old data, just save the new data
                    output_df=new_data_df

                # Save the data
                print('Saving benchmark data:')
                print(f'{output_df.head()}\n')
                output_df.to_csv(benchmark_datafile, index=False)

            j += 1
        i += 1

    return True


# generate points in latent space as input for the generator

def generate_latent_points(latent_dim, n_samples):

    # generate points in the latent space
    x_input=randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network
    x_input=x_input.reshape(n_samples, latent_dim)

    return x_input


# use the generator to generate n fake examples, with class labels

def generate_fake_samples(g_model, latent_dim, n_samples):

    # generate points in latent space
    x_input=generate_latent_points(latent_dim, n_samples)

    # predict outputs
    X=g_model.predict(x_input)

    # create 'fake' class labels (0)
    y=zeros((n_samples, 1))

    return X, y


# save 3x3 grid of generated images

def save_frame(g_model, latent_points, frame, image_output_dir):

    # create images from latent points
    images=g_model.predict(latent_points)

    # scale images into range 0.0, 1.0 for plotting as RGB
    images=(images + 1.0) / 2.0

    plot_dim=3
    fig=plt.figure(figsize=(13.3025, 13.3025), dpi=300)
    ax=[]

    for i in range(plot_dim * plot_dim):

        # create subplot and append to ax
        ax.append(fig.add_subplot(plot_dim, plot_dim, i+1))
        plt.imshow(images[i])
        plt.axis('off')

    # remove whitespace between plots
    plt.subplots_adjust(wspace=-0.019, hspace=0)

    # save plot to file
    filename=f'{image_output_dir}/frame{frame:07d}.jpg'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    return True


# Generates from model

def generate_specimen(g_model, latent_dim):

    # generate point in latent space
    x_input=generate_latent_points(latent_dim, 1)
    
    # predict outputs
    X=g_model.predict(x_input)

    return x_input, X


# Generates and saves model output and latent points

def save_specimen(g_model, latent_dim, filename):

    # Get image and latent points
    latent_points, specimen=generate_specimen(g_model, latent_dim)

    # Normalize
    specimen=(specimen + 1.0) / 2.0

    # Plot and save image
    image_filename=f'{filename}.jpg'
    plt.imsave(image_filename, specimen[0])

    # Save latent points as pickle for re-use
    with open(f'{filename}_latent_points.pkl', 'wb') as output:
        pickle.dump(latent_points, output)

    # Format and save latent points as text
    latent_points=np.reshape(latent_points, (10,10))

    with open(f'{filename}_latent_points.dat', 'w') as output:
        for row in latent_points:

            formatted_row=[]

            for i in row:
                formatted_row.append((f'{i:.4f}').rjust(7))

            formatted_row=','.join(formatted_row)
            output.write(f'{formatted_row}\n')

    # Save raw image data
    specimen=np.transpose(specimen[0], (2, 0, 1))

    with open(f'{filename}_gan_output.dat', 'w') as output:
        for channel in specimen:

            for row in channel:

                formatted_row=[]

                for i in row:
                    formatted_row.append(f'{i:.4f}')

                formatted_row=','.join(formatted_row)
                output.write(f'{formatted_row}\n')

            output.write('\n')

    return True


# Takes a model and a latent point, saves the model output

def save_specimen_from_latent_point(g_model, latent_point, filename):

    # Generate
    specimen=g_model.predict(latent_point)

    # Normalize
    specimen=(specimen + 1.0) / 2.0

    # Plot and save image
    image_filename=f'{filename}.jpg'
    plt.imsave(image_filename, specimen[0])

    return True