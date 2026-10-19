from pathlib import Path
import joblib
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

IMAGE_SIZE = (128, 128)

def extract_features(image_rgb):
    small = resize(image_rgb, (*IMAGE_SIZE, 3), anti_aliasing=True, preserve_range=True).astype(np.uint8)
    gray = rgb2gray(small)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

def load_classifier(path="models/wound_classifier.joblib"):
    return joblib.load(path)

def predict_image(model, image_rgb):
    features = extract_features(image_rgb).reshape(1, -1)
    label = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        score = float(np.max(model.predict_proba(features)[0]))
    else:
        score = 0.0
    return str(label), score
