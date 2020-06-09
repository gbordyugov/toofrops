import tensorflow as tf
from model import unet, unet2
from data import get_training_and_test_datasets

def dice_plus_x_entropy_loss(y_true, y_pred):
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    dice_loss = 1.0 - numerator / denominator
    x_entropy_loss  = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    return 1.0*dice_loss + 1.0*x_entropy_loss


def train(epochs=10, batch_size=5):
    model = unet()
    model.compile(optimizer='adam', loss=dice_plus_x_entropy_loss,
                  metrics=['accuracy'])

    train, test = get_training_and_test_datasets()

    # can consume to much memory on a larger dataset, but for 25 pics is OK
    train_size = len(list(train))
    test_size  = len(list(test))

    train_batched = train.batch(batch_size).repeat(epochs)
    test_batched  =  test.batch(batch_size).repeat(epochs)

    steps_per_epoch = train_size // batch_size
    validation_steps = train_size // batch_size

    model_history = model.fit(train_batched, epochs=epochs,
                              steps_per_epoch=steps_per_epoch)

    return model
