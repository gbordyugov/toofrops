import tensorflow as tf
import pathlib

IMAGE_DATA_URL = 'https://dida.do/assets/downloads/dida-test-task/dida_test_task.zip'
IMG_HEIGHT = 256
IMG_WIDTH = 256

def get_images_and_labels(url=IMAGE_DATA_URL):
    """ Download data, unpack it and return paths to images and labels. """
    download_path = tf.keras.utils.get_file(origin=url,
                                            fname='rooftops.zip',
                                            extract=True)
    path = pathlib.Path(download_path)
    parent = path.parents[0]

    images = parent / 'images'
    labels = parent / 'labels'

    assert images.is_dir()
    assert labels.is_dir()

    return images, labels


def match_images_with_labels(images, labels):
    """ Return matches (used for training) and non-matches between
    images and labels. Matches are returned as a dictionary, mapping
    filenames to tuples (image, label). Non-matches are returned
    as a dictionary mapping filenames to full paths. """

    assert images.is_dir()
    assert labels.is_dir()

    images_d = {f.parts[-1]: f for f in images.glob('*.png')}
    labels_d = {f.parts[-1]: f for f in labels.glob('*.png')}

    images_names = images_d.keys()
    labels_names = labels_d.keys()

    difference   = images_names - labels_names
    intersection = images_names & labels_names

    matches = {k: (images_d[k], labels_d[k]) for k in intersection}
    non_matches = {k : images_d[k] for k in difference}

    return matches, non_matches


def load_image_from_path(image_path, channels):
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_png(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def matches_to_dataset(matches):
    # TODO: fix the channels issue
    pairs = [ (load_image_from_path(image, channels=3),
               load_image_from_path(label, channels=1))
              for _, (image, label) in matches.items() ]
    return tf.data.Dataset.from_tensor_slices(pairs)
