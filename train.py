import tensorflow as tf

from model import unet
from data import get_training_dataset


def dice_plus_x_entropy_loss(y_true, y_pred):
    """ Calculate a combination of the Dice loss and the normal
    binary x-entropy loss. Code borrowed from
    https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    Experimental, not used in the "production" code.
    """
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    dice_loss = 1.0 - numerator / denominator
    x_entropy_loss  = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # adjust alpha between to mix the DC loss and XE loss
    alpha = 1.0
    return (1.0-alpha)*dice_loss + alpha*x_entropy_loss


def train(epochs=10, batch_size=5):
    model = unet()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    train = get_training_dataset()

    # can consume too much memory on a larger dataset, but for 25 pics is OK
    train_size = len(list(train))

    train_batched = train.batch(batch_size).repeat(epochs)

    steps_per_epoch = train_size // batch_size

    model_history = model.fit(train_batched, epochs=epochs,
                              steps_per_epoch=steps_per_epoch)

    return model

if '__main__' == __name__:
    model = train(30)
    model.save('model.tf')
