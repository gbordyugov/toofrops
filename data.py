import tensorflow as tf
import pathlib

IMAGE_DATA_URL = 'https://dida.do/assets/downloads/dida-test-task/dida_test_task.zip'

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
