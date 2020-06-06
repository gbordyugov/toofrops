import tensorflow as tf
import pathlib

IMAGE_DATA_URL = 'https://dida.do/assets/downloads/dida-test-task/dida_test_task.zip'

def get_images_and_labels(url = IMAGE_DATA_URL):
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
