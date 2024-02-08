import os
import shutil
import pathlib
import tensorflow as tf

def prep_output_dir(output_path, resume):

    # Check if the directory exists
    if pathlib.Path(output_path).is_dir():

        # It already exists, so empty it if we are not resuming
        if resume == False:
            for filename in os.listdir(output_path):
                file_path = os.path.join(output_path, filename)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    else:

        # It doesn't exist, so create it
        print(f'Creating {output_path}')
        pathlib.Path(output_path).mkdir()

    return True


def decode_img(img, image_dim):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [image_dim, image_dim])


def process_path(file_path, image_dim):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_dim)
    img = tf.image.convert_image_dtype(img, dtype=tf.float16, saturate=False)
    img = (img - 127.5) / 127.5

    return img


def prep_data(image_dir, batch_size, image_dim):
    data_dir = pathlib.Path(image_dir)
    image_count = len(list(data_dir.glob('*.jpg')))
    train_ds = tf.data.Dataset.list_files(
        f'{image_dir}/*.jpg', shuffle=True)

    train_ds = train_ds.shuffle(image_count, reshuffle_each_iteration=True)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(lambda x: process_path(
        x, image_dim), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    return train_ds, image_count
