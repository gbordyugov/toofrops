import tensorflow as tf
import matplotlib.pyplot as plt

from data import get_unmatches

def make_one_prediction(model, input):
    """ Return a binary prediction using model on input. """
    pred = model(input[tf.newaxis,:,:,:], training=False).numpy()

    pred = pred[0,:,:,0]

    pred[pred >= 0.5] = 1.0
    pred[pred <  0.5] = 0.0

    return pred

def predict_unmatches(model, filename):
    unmatches = get_unmatches()
    preds = [make_one_prediction(model, u) for u in unmatches]
    fig, axes = plt.subplots(5, 2, figsize=(8, 18))

    for (ax1, ax2), unmatch, pred in zip(axes, unmatches, preds):
        ax1.imshow(unmatch)
        ax2.imshow(pred)
        ax1.axis('off')
        ax2.axis('off')

    fig.savefig(filename)


if '__main__' == __name__:
    model = tf.keras.models.load_model('model.tf')
    predict_unmatches(model, 'predictions.png')
