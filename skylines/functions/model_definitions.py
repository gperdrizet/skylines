from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose

from keras.optimizers import Adam


def define_discriminator(learning_rate, in_shape=(1024, 1024, 3)):

    model=Sequential()

    # Input
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    # Downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    opt=Adam(learning_rate=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



# Define the standalone generator model

def define_generator(latent_dim):

    model=Sequential()

    # Foundation for 32x32 image
    n_nodes=1024 * 32 * 32
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((32, 32, 1024)))

    # Upsample to 64x64
    model.add(Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample to 128x128
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample to 256x256
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample to 512x512
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample to 1024x1024
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(16, (6, 6), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

    return model


# Define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model, learning_rate):

    # Make weights in the discriminator not trainable
    d_model.trainable=False

    # Connect them
    model=Sequential()

    # Add generator
    model.add(g_model)

    # Add the discriminator
    model.add(d_model)

    # Compile model
    opt=Adam(learning_rate=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model
