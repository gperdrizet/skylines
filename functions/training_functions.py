from numpy import zeros
from numpy import ones
from numpy.random import randn
import matplotlib.pyplot as plt

# generate points in latent space as input for the generator


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    # x_input = tf.convert_to_tensor(x_input, dtype=tf.float16)

    return x_input

# use the generator to generate n fake examples, with class labels


def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

# save 3x3 grid of generated images


def save_frame(g_model, latent_points, frame, image_output_dir):
    # create images from latent points
    images = g_model.predict(latent_points)
    # scale images into range 0.0, 1.0 for plotting as RGB
    images = (images + 1.0) / 2.0

    plot_dim = 3

    fig = plt.figure(figsize=(13.3025, 13.3025), dpi=300)
    ax = []

    for i in range(plot_dim * plot_dim):
        # create subplot and append to ax
        ax.append(fig.add_subplot(plot_dim, plot_dim, i+1))
        plt.imshow(images[i])
        plt.axis('off')

    # remove whitespace between plots
    plt.subplots_adjust(wspace=-0.019, hspace=0)

    # save plot to file
    filename = f'{image_output_dir}/frame{frame:07d}.jpg'
    #filename = './gan_output/frame%07d.jpg' % (frame)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    frame += 1

    return frame

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
    image_output_dir
):

    bat_per_epo = int((image_count) / n_batch)

    # loop on epochs
    for i in range(n_epochs):
        iterator = iter(dataset)

        # loop on batches
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real = iterator.get_next()
            y_real = ones((n_batch, 1))
            # train descriminator on real samples
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(
                g_model, latent_dim, n_batch)
            X_fake = X_fake
            # train descriminator on fake samples from generator
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # train the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # summarize loss on this batch
            print(f'{project_name}-{frame}: d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}')

            frame = save_frame(g_model, latent_points, frame, image_output_dir)

            if frame % checkpoint_save_frequency == 0:
                # filename = './training_checkpoints/generator_model_f%07d.h5' % (
                #     frame)
                g_model.save(f'{model_checkpoint_dir}/generator_model_f{frame:07d}.h5')
                # filename = './training_checkpoints/discriminator_model_f%07d.h5' % (
                #     frame)
                d_model.save(f'{model_checkpoint_dir}/discriminator_model_f{frame:07d}.h5')
                # filename = './training_checkpoints/gan_model_f%07d.h5' % (
                #     frame)
                gan_model.save(f'{model_checkpoint_dir}/gann_model_f{frame:07d}.h5')

            j += 1
        i += 1

def generate_specimen(g_model, latent_dim):
    # generate point in latent space
    x_input = generate_latent_points(latent_dim, 1)
    # predict outputs
    X = g_model.predict(x_input)
    return X


def save_specimen(g_model, latent_dim, filename):

    specimen = generate_specimen(g_model, latent_dim, )
    specimen = (specimen + 1.0) / 2.0

    plt.imsave(filename, specimen[0])