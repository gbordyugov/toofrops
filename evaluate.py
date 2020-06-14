import tensorflow as tf
from numpy import array

from model import unet
from data import get_dataset

def train_and_validate(epochs=10, batch_size=5, test_batch_size=1):
    """ Train and validate model using `test_batch_size` batches to
    calculate test accuracy. Return test accuracy from the last epoch.
    """
    model = unet()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    ds = get_dataset()
    ds = ds.shuffle(len(list(ds)))
    ds = ds.batch(batch_size)

    test_ds = ds.take(test_batch_size)
    train_ds = ds.skip(test_batch_size)

    history = model.fit(train_ds, epochs=epochs,
                        validation_data=test_ds)

    return history.history['val_accuracy'][-1]


def evaluate(num_rounds=10):
    """ Run `num_rounds` evaluation steps, collect the validation
    accuracies and return their mean."""
    accs = [train_and_validate() for _ in range(num_rounds)]
    return array(accs).mean()


if '__main__' == __name__:
    accuracy = evaluate()
    print('Average validation accuracy:', accuracy)
