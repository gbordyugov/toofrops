import tensorflow as tf

def make_one_prediction(model, input):
    pred = model(input[tf.newaxis,:,:,:], training=False).numpy()

    pred = pred[0,:,:,0]

    pred[pred >= 0.5] = 1.0
    pred[pred <  0.5] = 0.0

    return pred
