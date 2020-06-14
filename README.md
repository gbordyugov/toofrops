# Rooftops project

## How to run the code

Running

```
pip install -r requirements.txt
python train.py                            # takes a couple of minutes
python inference.py                        # takes a couple of seconds
```

(preferrably in a fresh Python 3.7 virtual environment) will produce
the file `predictions.png` that shows the predicted masks for the five
pictures without labels. In addition, the trained model is saved as
`model.tf` Judging by the the training data only (see the paragraph
below about model evaluation using cross-validation), an accuracy of
nearly 91% is achieved.

I am also including
[predictions-reference.png](predictions-reference.png) such that you
can see the results without having to run the whole training/inference
pipeline.

## Validation and performance evalution of the model

Running

```
python evaluate.py
```

will run an automatic performance evaluation of the model.

By default, in every of 10 rounds, the whole 25 labelled images would
be randomly split into a train dataset with 20 pictures and a test
detaset with 5 pictures. The model would be trained on those 20
training pictures over 10 epochs, and the validation set of 5 pictures
would be used to calculated the loss and accuracy. The validation loss
and accuracy after the last training epoch in every of 10 rounds would
be averaged over and their mean values will be reported at the end of
the script runtime. Here, I'm reporting the current value of

```
```

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

The differences of my code from the above tutorial are:

- The problem at hand requires just a binary classifier unlike the
  three-way classifier in the tutorial.
- Because of the binary classification, I use a single output channel
  and a somewhat simpler loss function (but still a cross-entropy
  loss).
- My networ outputs classification probabilities rather then
  log-probs, which is achieved by having a sigmoid activation in the
  last layer, the reason being just two classification classes.
- I experimented with the Dice loss plus a weighted combination of the
  cross-entropy and Dice loss, but couldn't find any substantial
  benefits of the Dice loss.

I experimented with a hand-crafted U-Net architecture, too (not
included of the code), but it delivered substantially worse results
than the network with a pre-trained encoder.


## Possible improvements

- Having more data would help, as always in machine learning.
- Engineering a more sophisticated loss function, for instance, one
  penalsing false positives more than false negatives (additional
  research would be needed).
- The pictures seem to have different exposure, some of them being
  seemingly taken on overcast days. A smarter data normalisation
  strategy cold help to mitigate that.
- Another idea for improving the loss function would be to apply some
  domain-specific knowledge about the shape of the predicted masks.
  For example, one could penalise the degree of non-rectangleness of
  the generated masks. Or calculate the number of predicted rooftops
  (as the number of connected positive regions in the predicted mask)
  and penalise its discrepancy from the ground truth.
- I went with the vanilla settings of the Adam optimiser, it's
  probable that it would be possible to squeeze another performance
  improvement by tweaking it a bit.
