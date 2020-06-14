import tensorflow as tf
import tensorflow_addons as tfa
from pathlib import Path
from random import randint

from constants import IMG_WIDTH, IMG_HEIGHT

IMAGE_DATA_URL = 'https://dida.do/assets/downloads/dida-test-task/dida_test_task.zip'

def download_images_and_labels(url=IMAGE_DATA_URL):
    """ Download data, unpack it and return paths to images and labels. """
    download_path = tf.keras.utils.get_file(origin=url,
                                            fname='rooftops.zip',
                                            extract=True)
    path = Path(download_path)
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
    unmatches = {k : images_d[k] for k in difference}

    return matches, unmatches


def load_png(image_path, channels):
    """ Load, normalize, and resize an image from the given path."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_png(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def convert_matches_to_dataset(matches):
    """ From matches, being a dictionary containing paths to the
    images and the corresponding labels, load data and return it as a
    tf Dataset."""
    Dataset = tf.data.Dataset

    pairs = [(load_png(image_path, channels=3),
              load_png(label_path, channels=1))
             for _, (image_path, label_path) in matches.items()]

    images_ds = Dataset.from_tensor_slices([i for i, l in pairs])
    labels_ds = Dataset.from_tensor_slices([l for i, l in pairs])

    return Dataset.zip((images_ds, labels_ds))


def load_unmatches(unmatches):
    """ Load unlabelled images from the paths specified by the
    dictionary `unmatches` and return them as a list. """
    return [load_png(path, channels=3) for _, path in unmatches.items()]


def get_unmatches():
    """ The top-level function for loading unmatches (= non-labelled
    images). """
    images, labels = download_images_and_labels()
    _, unmatches = match_images_with_labels(images, labels)
    return load_unmatches(unmatches)


def augment_by_rotations(ds, no_rotations=0):
    """ Explode the dataset by generating additional rotations. """
    Dataset = tf.data.Dataset
    def rotations(image, label):
        """ Return a dataset of random rotations of image and labels,
        both rotated in parallel. Include the trivial rotation, too."""

        angles = [0] + [randint(0, 360) for _ in range(no_rotations)]

        rotated_images = [tfa.image.rotate(image, a) for a in angles]
        rotated_labels = [tfa.image.rotate(label, a) for a in angles]

        images = Dataset.from_tensor_slices(rotated_images)
        labels = Dataset.from_tensor_slices(rotated_labels)

        return Dataset.zip((images, labels))

    return ds.flat_map(rotations)

def map_image_by(ds, f):
    """ Maps by f the image part of the (image, label) tuples in the
    dataset ds."""

    def mapper(image, label):
        image = f(image)
        return image, label

    return ds.map(mapper)


def get_dataset(rotations=4, shuffle_size=100):
    """ The top-level function for loading matches (= labelled images)
    and returning them as a dataset. """
    images, labels = download_images_and_labels()
    matches, _ = match_images_with_labels(images, labels)
    ds = convert_matches_to_dataset(matches)

    #
    # With a pre-trained encoder, both seem quite useless so far.
    #
    # ds = augment_by_rotations(ds, rotations)
    # ds = map_image_by(ds,
    #                   lambda x: tf.image.random_jpeg_quality(x, 30, 70))

    return ds
