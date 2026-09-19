import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "maize_cnn.h5"
TEST_DIR = ROOT / "data" / "test"
IMG_SIZE = (128, 128)


def load_test_set():
    classes = sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
    images, labels = [], []
    for index, name in enumerate(classes):
        for path in sorted((TEST_DIR / name).iterdir()):
            img = keras.utils.load_img(path, target_size=IMG_SIZE, interpolation="nearest")
            images.append(keras.utils.img_to_array(img) / 255.0)
            labels.append(index)
    return classes, np.array(images), np.array(labels)


def main():
    model = keras.models.load_model(MODEL_PATH)
    classes, images, labels = load_test_set()
    predicted = model.predict(images, verbose=0).argmax(axis=1)

    print(f"Test images: {len(labels)}")
    print(f"Accuracy:    {(predicted == labels).mean():.4f}\n")
    print(f"{'class':16s}{'precision':>10s}{'recall':>8s}{'images':>8s}")
    for index, name in enumerate(classes):
        hits = ((predicted == index) & (labels == index)).sum()
        precision = hits / max((predicted == index).sum(), 1)
        recall = hits / (labels == index).sum()
        print(f"{name:16s}{precision:10.2f}{recall:8.2f}{(labels == index).sum():8d}")


if __name__ == "__main__":
    main()
