import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2DTranspose

from constants import IMG_WIDTH, IMG_HEIGHT

#
# Most of this model code is borrowed from
# https://www.tensorflow.org/tutorials/images/segmentation
#

def unet():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    down_stack = Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3),
    ]

    inputs = Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    x = inputs

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

        last = Conv2DTranspose(1, 3, strides=2, activation='sigmoid',
                               padding='same')

    x = last(x)

    return Model(inputs=inputs, outputs=x)
