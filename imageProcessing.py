import cv2
import numpy as np

def preprocess_image(image_rgb, size=(512, 512)):
    """Return a fixed-size RGB image and a contrast-enhanced RGB image."""
    resized = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2RGB)
    return resized, enhanced

def segment_wound_baseline(image_rgb):
    """
    Basic experimental threshold-based mask.
    This is NOT a reliable wound segmentation model.
    Replace/evaluate with a trained segmentation model using annotated masks.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Broad red/pink color ranges; illumination and skin tones strongly affect this.
    low_red1 = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([15, 255, 255]))
    low_red2 = cv2.inRange(hsv, np.array([160, 25, 35]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(low_red1, low_red2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def image_measurements(image_rgb, mask):
    """Compute image-only measurements; pixel fraction is not physical area."""
    h, w = image_rgb.shape[:2]
    fraction = float(np.count_nonzero(mask)) / float(mask.size) * 100.0
    mean_rgb = np.mean(image_rgb, axis=(0, 1)).round(2).tolist()
    return {
        "width_px": int(w),
        "height_px": int(h),
        "mask_fraction_pct": round(fraction, 4),
        "mean_rgb": mean_rgb
    }
