# Maize Disease CNN

A small convolutional neural network that classifies maize leaf photos into five classes, and a Streamlit app that uses it to give farmers basic disease guidance in English or Swahili.

Classes:

- Blight
- Common Rust
- Gray Leaf Spot
- Healthy
- Other (anything that is not a maize leaf: other plants' leaves, animals, scenery)

The Other class exists so the app can reject photos that are not maize leaves instead of forcing them into one of the disease classes.

## Layout

```
app.py                  Streamlit app
content/en.json         App text and disease information, English
content/sw.json         App text and disease information, Swahili
assets/style.css        App styling
models/maize_cnn.h5     Trained model used by the app
train.ipynb             Training notebook
scripts/evaluate.py     Scores a model against the test set
data/train/<class>/     Training images (the notebook splits these 80/20 into train and validation)
data/test/<class>/      Held out test images, 172 per class (169 for Other)
```

## Setup

Tested on Python 3.12.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For training, install `requirements-train.txt` instead. It adds Jupyter, matplotlib, seaborn and scikit-learn.

## Run the app

```
streamlit run app.py
```

Upload a leaf photo on the Upload page and the app switches to Results. The language switch is at the top right. Any image in `data/test/` works as a sample.

The app rejects an image in two cases: when the model predicts Other, and when the top prediction is below 70% confidence. The threshold is `CONFIDENCE_THRESHOLD` in `app.py`.

## Evaluate

```
python scripts/evaluate.py
python scripts/evaluate.py path/to/other_model.h5
```

Current model on the test set (857 images):

| class          | precision | recall |
|----------------|-----------|--------|
| Blight         | 0.96      | 0.71   |
| Common Rust    | 0.95      | 1.00   |
| Gray Leaf Spot | 0.67      | 0.97   |
| Healthy        | 0.93      | 0.99   |
| Other          | 0.98      | 0.67   |

Overall accuracy is 86.8%.

## Train

Open `train.ipynb` and run the cells in order. It builds a three block CNN on 128x128 images with light augmentation, trains for 25 epochs, prints a classification report and confusion matrix, and saves the result to `models/maize_cnn.h5`.

The class order comes from the alphabetical order of the folders in `data/train/`. `CLASS_LABELS` in `app.py` must follow the same order. If you add, remove or rename a class folder, update that list.

The saved notebook outputs come from the original training run, before the dataset cleanup described below. That is why the first cell reports 4019/1003/860 images rather than the current counts.

## Dataset notes

- The original source of the images was not recorded. The maize photos look like a public Kaggle maize leaf disease set. The Other class mixes in images from the Kaggle cats and dogs set, scenery photos and leaves from other plants. Check the licences before using the data outside internal or learning work.
- Some files have an `augmented_` prefix. These were already augmented before training and are not produced by the notebook.
- Removed during cleanup: 60 Other images that also appeared in the test set, 8 exact duplicate copies, 2 images that appeared under both Blight and Gray Leaf Spot (both copies dropped), and 3 duplicate copies inside the test set.

## Known limitations

- The model overfits. Training accuracy reaches about 90% while validation accuracy ends near 70%, and validation loss rises in the later epochs. Early stopping, more augmentation, or transfer learning from a pretrained backbone such as MobileNetV2 or EfficientNet are the obvious next steps.
- Blight recall is 0.71. Of 172 blight test images, 48 are predicted as Gray Leaf Spot.
- Other recall is 0.67. Of 169 Other test images, 35 are predicted as Gray Leaf Spot, 12 as Healthy (all leaves from other plants) and 8 as Common Rust. Rejecting non-maize photos is not reliable, especially for photos of other plants' leaves.
- The model has not been retrained since the dataset cleanup.
- The Resources links in the content files point to the FAO homepage, not to specific disease guides.
- The model is saved in the legacy HDF5 format. Newer Keras versions prefer `.keras`.
