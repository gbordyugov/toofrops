import tensorflow as tf
from model import unet
from data import get_training_and_test_datasets

def train(epochs=20, batch_size=10):
    model = unet(3)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
