import tensorflow as tf
from model import unet
from data import get_training_and_test_datasets

def dice_loss(y_true, y_pred):
  numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
  return 1.0 - numerator / denominator

def train(epochs=10, batch_size=5):
    model = unet(2)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # loss=dice_loss,
                  metrics=['accuracy'])

    train, test = get_training_and_test_datasets()
    train_size = len(list(train))
    test_size  = len(list(test))

    train_batched = train.batch(batch_size).repeat(epochs)
    test_batched  =  test.batch(batch_size).repeat(epochs)

    steps_per_epoch = train_size // batch_size
    validation_steps = train_size // batch_size

    model_history = model.fit(train_batched.repeat(epochs), epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_data=test_batched)
    return model
