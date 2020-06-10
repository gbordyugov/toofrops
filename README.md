# Rooftops project

## How to run the code

Running

```
pip install -r requirements.txt
python train.py                            # takes a couple of minutes
python inference.py                        # takes a couple of seconds
```

(preferrably in a fresh Python 3.7 virtual environment) will produce
`predictions.png` that shows the predicted masks for the five pictures
without labels. In addition, the trained model is saved as `model.tf`
Judging by the the training data only (see the remark below on the
absence of the validation/test data), an accuracy of nearly 91% is
achieved.

I am also including
[predictions-reference.png](predictions-reference.png) such that you
can see the results without having to run the whole training/inference
pipeline.

## Details of implementation

### Data handling

`data.py` downloads the data and creates a training dataset. Due to
the sparseness of the data, I decided to use all 25 labelled images
for training (no validation/test set) and to verify the quality of
training by the training accuracy plus visual inspection of the
predictions for the unlabelled data.

I experimented with augmenting the data by rotations, but didn't find
it beneficial. A possible explanation would be that I'm using a
pre-trained encoder part of the network which is already doing quite a
good job of robustly recongnising features in the input data.

### Model description

I mostly followed the ideas from
[this](https://www.tensorflow.org/tutorials/images/segmentation)
tutorial. The main point is to utilize a U-Net architecture with a
pre-trained encoder part, taken from the MobileNetV2 model, which has
a total of 155 single layers.

I experimented with a hand-crafted U-Net architecture, too (not
included of the code), but it delivered substantially worse results
than the network with a pre-trained encoder.

The differences of my code from the above tutorial are:

- Just binary classification unlike the three-classes in the tutorial.
- Because of the binary classification, I use a single output channel
  and a somewhat simpler loss function (but still a cross-entropy
  loss).
- My networ outputs classification probabilities rather then
  log-probs, which is achieved by having a sigmoid activation in the
  last layer, the reason being just two classification classes.
- I experimented with the Dice loss plus a weighted combination of the
  cross-entropy and Dice loss, but couldn't find any substantial
  benefits of the Dice loss.

### Possible improvements

Having more data would help, as always in machine learning.

The pictures seem to have different exposure, some of them being
seemingly taken on overcast days. A smarter data normalisation strategy
cold help to mitigate that.
