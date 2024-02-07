import sys
import PIL
from PIL import Image
from pathlib import Path
#from shutil import copyfile
import config

count = 0

for image in Path(f'{config.RAW_IMAGE_DIR}/').rglob('*.jpg'):

    # load and resize image
    try:
        img = Image.open(image)
        img = img.resize((config.IMAGE_DIM, config.IMAGE_DIM))
    
    except KeyboardInterrupt:
        print('Caught keyboard interupt.')
        sys.exit(1)

    except:
        error = sys.exc_info()
        print(f'Could not load image: {error}')

    # save unflipped version
    try:
        destination = f'{config.PROCESSED_IMAGE_DIR}/{count:06}.jpg'
        img.save(destination)
        count += 1
    
    except KeyboardInterrupt:
        print('Caught keyboard interupt.')
        sys.exit(1)

    except:
        error = sys.exc_info()
        print(f'Could save orignal image: {error}')

    # flip and save
    try:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        destination = f'{config.PROCESSED_IMAGE_DIR}/{count:06}.jpg'
        img.save(destination)
        count += 1

    except KeyboardInterrupt:
        print('Caught keyboard interupt.')
        sys.exit(1)

    except:
        error = sys.exc_info()
        print(f'Could save flipped image: {error}')