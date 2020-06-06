import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2DTranspose

from constants import IMG_WIDTH, IMG_HEIGHT

def unet(output_channels):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

        # This is the last layer of the model
        last = Conv2DTranspose(output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128

    x = last(x)

    return Model(inputs=inputs, outputs=x)
