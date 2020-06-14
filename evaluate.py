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
                        validation_data=test_ds).history

    loss = history['val_loss'][-1]
    acc  = history['val_accuracy'][-1]

    return loss, acc


def evaluate(num_rounds=10):
    """ Run `num_rounds` evaluation steps, collect the validation
    accuracies and return their mean."""
    losses_and_accs = [train_and_validate(30) for _ in range(num_rounds)]
    return array(losses_and_accs).mean(axis=0)


if '__main__' == __name__:
    loss, accuracy = evaluate()
    print('Average validation loss:', loss)
    print('Average validation accuracy:', accuracy)
