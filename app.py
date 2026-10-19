import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib

from PIL import Image
from pathlib import Path
from skimage.feature import hog

from imageProcessing import (
    preprocess_image,
    segment_wound_baseline,
    image_measurements
)

from database import (
    init_db,
    save_visit,
    get_visits,
    get_patient_codes,
    get_all_visits
)

from ModelUtils import load_classifier, predict_image


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI-Based Wound Assessment",
    page_icon="🩹",
    layout="wide"
)

st.title("AI-Based Wound Assessment")
st.subheader("Patient-wise Wound Assessment Dashboard")

st.write(
    "An academic prototype for wound image processing, "
    "experimental classification, image measurements, "
    "and patient-wise visit history."
)

st.warning(
    "Research prototype only. Predictions and segmentation "
    "are not clinically validated. Do not use this application "
    "for diagnosis or treatment decisions."
)


# =====================================================
# 2. MODEL PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

BINARY_MODEL_PATH = (
    BASE_DIR / "models" / "wound_classifier_binary.joblib"
)

TYPE_MODEL_PATH = (
    BASE_DIR / "models" / "wound_type_classifier.joblib"
)


# =====================================================
# 3. INITIALIZE DATABASE
# =====================================================

database_ready = False

try:
    init_db()
    database_ready = True

except Exception as exc:
    st.error(f"Database initialization failed: {exc}")


# =====================================================
# 4. LOAD BINARY CLASSIFICATION MODEL
# =====================================================

@st.cache_resource
def load_binary_model(model_path):
    return load_classifier(str(model_path))


binary_model = None

if BINARY_MODEL_PATH.exists():
    try:
        binary_model = load_binary_model(BINARY_MODEL_PATH)

    except Exception as exc:
        st.error(f"Could not load binary model: {exc}")

else:
    st.error(
        f"Binary model not found: {BINARY_MODEL_PATH}"
    )


# =====================================================
# 5. LOAD WOUND-TYPE MODEL
# =====================================================

@st.cache_resource
def load_type_model(model_path):

    saved_object = joblib.load(model_path)

    if isinstance(saved_object, dict):

        model = saved_object["model"]

        classes = saved_object.get(
            "classes",
            {
                "D": "Diabetic",
                "S": "Surgical",
                "V": "Venous"
            }
        )

    else:

        model = saved_object

        classes = {
            "D": "Diabetic",
            "S": "Surgical",
            "V": "Venous"
        }

    return model, classes


type_model = None
type_classes = {}

if TYPE_MODEL_PATH.exists():

    try:
        type_model, type_classes = load_type_model(
            TYPE_MODEL_PATH
        )

    except Exception as exc:
        st.error(f"Could not load wound-type model: {exc}")

else:
    st.error(
        f"Wound-type model not found: {TYPE_MODEL_PATH}"
    )


# =====================================================
# 6. WOUND-TYPE FEATURE EXTRACTION
# =====================================================

def extract_wound_type_features(image_rgb):

    image_rgb = np.asarray(
        image_rgb,
        dtype=np.uint8
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    image_bgr = cv2.resize(
        image_bgr,
        (128, 128)
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


def predict_wound_type(model, image_rgb):

    features = extract_wound_type_features(image_rgb)

    prediction = str(
        model.predict(features)[0]
    )

    return prediction


# =====================================================
# 7. PATIENT REGISTRATION AND SELECTION
# =====================================================

st.divider()
st.header("1. Patient Record")

try:

    if database_ready:
        existing_patient_codes = get_patient_codes()
    else:
        existing_patient_codes = []

except Exception as exc:

    existing_patient_codes = []

    st.error(
        f"Could not load patient records: {exc}"
    )


patient_mode = st.radio(
    "Choose an option",
    [
        "Existing patient",
        "Register new patient"
    ],
    horizontal=True
)


if patient_mode == "Existing patient":

    if existing_patient_codes:

        patient_code = st.selectbox(
            "Select Patient ID",
            options=existing_patient_codes
        )

    else:

        patient_code = ""

        st.info(
            "No patients have been registered yet. "
            "Choose 'Register new patient' to create the first record."
        )


else:

    patient_code = st.text_input(
        "Enter a new anonymous Patient ID",
        max_chars=40,
        placeholder="Example: W001",
        help=(
            "Use an anonymous ID. Do not enter names "
            "or other identifying details."
        )
    ).strip()

    if patient_code:

        if patient_code in existing_patient_codes:

            st.error(
                "This Patient ID already exists. "
                "Select 'Existing patient' to add a follow-up visit."
            )

        else:

            st.success(
                f"New Patient ID entered: {patient_code}"
            )


visit_date = st.date_input(
    "Visit Date"
)


# =====================================================
# 8. IMAGE UPLOAD
# =====================================================

st.divider()
st.header("2. Upload Wound Image")

uploaded_file = st.file_uploader(
    "Choose a wound image",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# 9. PROCESS IMAGE AND MAKE PREDICTIONS
# =====================================================

if uploaded_file is not None:

    try:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        original_array = np.asarray(
            original_image
        )

        # Image preprocessing
        resized, enhanced = preprocess_image(
            original_array
        )

        # Experimental segmentation
        mask = segment_wound_baseline(
            enhanced
        )

        # Image measurements
        measurements = image_measurements(
            enhanced,
            mask
        )

        # -------------------------------------------------
        # IMAGE PROCESSING RESULTS
        # -------------------------------------------------

        st.divider()
        st.header("3. Image Processing")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.subheader("Enhanced Image")

            st.image(
                enhanced,
                use_container_width=True
            )

        st.subheader(
            "Experimental Segmentation Mask"
        )

        st.image(
            mask,
            caption=(
                "Threshold-based region proposal. "
                "This is not a clinically validated wound boundary."
            ),
            use_container_width=True
        )

        # -------------------------------------------------
        # IMAGE MEASUREMENTS
        # -------------------------------------------------

        st.divider()
        st.header("4. Image Measurements")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Image Width",
            f"{measurements['width_px']} px"
        )

        col2.metric(
            "Image Height",
            f"{measurements['height_px']} px"
        )

        col3.metric(
            "Mask Pixel Fraction",
            f"{measurements['mask_fraction_pct']:.2f}%"
        )

        st.write(
            "**Mean RGB values:**",
            measurements["mean_rgb"]
        )

        st.caption(
            "These measurements are derived from the image. "
            "Pixels do not represent physical wound dimensions "
            "without a calibration reference."
        )

        # -------------------------------------------------
        # BINARY CLASSIFICATION
        # -------------------------------------------------

        st.divider()
        st.header("5. Binary Classification")

        binary_label = None
        wound_type_label = None

        if binary_model is not None:

            try:

                label, score = predict_image(
                    binary_model,
                    enhanced
                )

                label_mapping = {
                    "Nomal": "Normal",
                    "Normal": "Normal",
                    "wound_main": "Wound"
                }

                binary_label = label_mapping.get(
                    str(label),
                    str(label)
                )

                st.success(
                    f"Predicted class: {binary_label}"
                )

                if score is not None and score > 0:

                    st.write(
                        f"Model score: {score:.2%}"
                    )

                st.caption(
                    "The model score is not necessarily "
                    "a calibrated probability."
                )

            except Exception as exc:

                st.error(
                    f"Binary prediction failed: {exc}"
                )

        else:

            st.error(
                "The binary classification model is unavailable."
            )

        # -------------------------------------------------
        # WOUND-TYPE CLASSIFICATION
        # -------------------------------------------------

        st.divider()
        st.header("6. Experimental Wound-Type Classification")

        if binary_label == "Normal":

            st.info(
                "Wound-type classification was skipped "
                "because the binary model predicted Normal."
            )

        elif binary_label == "Wound":

            if type_model is not None:

                try:

                    predicted_code = predict_wound_type(
                        type_model,
                        original_array
                    )

                    fallback_classes = {
                        "D": "Diabetic",
                        "S": "Surgical",
                        "V": "Venous"
                    }

                    wound_type_label = type_classes.get(
                        predicted_code,
                        fallback_classes.get(
                            predicted_code,
                            predicted_code
                        )
                    )

                    st.success(
                        "Predicted wound category: "
                        f"{wound_type_label}"
                    )

                    st.caption(
                        "This experimental classification is not "
                        "a diagnosis. Pressure wounds are not included "
                        "in this model."
                    )

                except Exception as exc:

                    st.error(
                        f"Wound-type prediction failed: {exc}"
                    )

            else:

                st.info(
                    "The wound-type model is unavailable."
                )

        else:

            st.info(
                "Wound-type classification was skipped because "
                "the binary prediction is unavailable or unrecognized."
            )

        # -------------------------------------------------
        # SAVE VISIT
        # -------------------------------------------------

        st.divider()
        st.header("7. Save Assessment")

        st.write(
            f"**Patient ID:** {patient_code or 'Not selected'}"
        )

        st.write(
            f"**Visit date:** {visit_date}"
        )

        if not database_ready:

            st.error(
                "The database is unavailable. "
                "The visit cannot be saved."
            )

        elif st.button(
            "Save Visit",
            type="primary"
        ):

            if not patient_code:

                st.error(
                    "Select or enter a Patient ID first."
                )

            elif (
                patient_mode == "Register new patient"
                and patient_code in existing_patient_codes
            ):

                st.error(
                    "This Patient ID already exists. "
                    "Choose 'Existing patient' to add a follow-up visit."
                )

            elif binary_label is None:

                st.error(
                    "A binary prediction is required before saving."
                )

            else:

                try:

                    predicted_result = binary_label

                    if wound_type_label:

                        predicted_result += (
                            f" - {wound_type_label}"
                        )

                    save_visit(
                        patient_code=patient_code,
                        visit_date=str(visit_date),
                        image_name=uploaded_file.name,
                        measurements=measurements,
                        predicted_label=predicted_result
                    )

                    st.success(
                        f"Visit saved successfully for {patient_code}."
                    )

                    st.info(
                        "A new visit record was added. "
                        "Previously saved visits were not overwritten."
                    )

                except Exception as exc:

                    st.error(
                        f"Could not save the visit: {exc}"
                    )

    except Exception as exc:

        st.error(
            f"Could not process this image: {exc}"
        )


# =====================================================
# 10. PATIENT-WISE DASHBOARD
# =====================================================

st.divider()
st.header("8. Patient Dashboard and Visit History")

if database_ready and patient_code:

    try:

        visits = get_visits(
            patient_code
        )

        if not visits.empty:

            history_df = visits.copy()

            history_df["visit_date"] = pd.to_datetime(
                history_df["visit_date"],
                errors="coerce"
            )

            history_df = history_df.sort_values(
                ["visit_date", "id"]
            )

            st.subheader(
                f"Patient ID: {patient_code}"
            )

            # ---------------------------------------------
            # PATIENT SUMMARY
            # ---------------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Saved Visits",
                len(history_df)
            )

            valid_dates = (
                history_df["visit_date"].dropna()
            )

            if not valid_dates.empty:

                col2.metric(
                    "First Visit",
                    valid_dates.min().strftime("%d %b %Y")
                )

                col3.metric(
                    "Latest Visit",
                    valid_dates.max().strftime("%d %b %Y")
                )

            else:

                col2.metric(
                    "First Visit",
                    "—"
                )

                col3.metric(
                    "Latest Visit",
                    "—"
                )

            # ---------------------------------------------
            # VISIT HISTORY TABLE
            # ---------------------------------------------

            st.subheader(
                "Assessment History"
            )

            display_df = history_df.copy()

            display_df["visit_date"] = (
                display_df["visit_date"]
                .dt.strftime("%Y-%m-%d")
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # ---------------------------------------------
            # PROGRESS CHART
            # ---------------------------------------------

            chart_df = history_df[
                [
                    "visit_date",
                    "mask_fraction_pct"
                ]
            ].copy()

            chart_df["mask_fraction_pct"] = pd.to_numeric(
                chart_df["mask_fraction_pct"],
                errors="coerce"
            )

            chart_df = chart_df.dropna()

            if len(chart_df) >= 2:

                chart_df = chart_df.set_index(
                    "visit_date"
                )

                st.subheader(
                    "Mask Pixel Fraction Across Visits"
                )

                st.line_chart(
                    chart_df["mask_fraction_pct"]
                )

                first_value = float(
                    chart_df["mask_fraction_pct"].iloc[0]
                )

                latest_value = float(
                    chart_df["mask_fraction_pct"].iloc[-1]
                )

                change = latest_value - first_value

                st.write(
                    "Difference between the first and latest "
                    f"saved mask pixel fractions: **{change:+.2f} "
                    "percentage points**"
                )

                st.caption(
                    "This chart shows an image-derived mask fraction. "
                    "It is not a physical wound-area measurement and "
                    "does not establish whether a wound has healed "
                    "or worsened."
                )

            elif len(chart_df) == 1:

                st.info(
                    "The progress chart will appear after at least "
                    "two visits have been saved for this patient."
                )

        else:

            st.info(
                "No visits are saved for this patient yet. "
                "Upload an image and save the first assessment."
            )

    except Exception as exc:

        st.error(
            f"Could not retrieve patient history: {exc}"
        )

elif not patient_code:

    st.info(
        "Select an existing patient or enter a new anonymous "
        "Patient ID to view its history."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "Technologies: Python, Streamlit, OpenCV, HOG feature extraction, "
    "classification models, threshold-based segmentation, and SQLite."
)

# =====================================================
# 12. ALL PATIENTS DASHBOARD
# =====================================================

st.divider()
st.header("9. All Patients Records")

if database_ready:
    try:
        all_visits = get_all_visits()
        all_patient_codes = get_patient_codes()

        col1, col2 = st.columns(2)
        col1.metric("Total Patients", len(all_patient_codes))
        col2.metric("Total Visits", len(all_visits))

        if all_visits.empty:
            st.info("No patient visits have been saved yet.")
        else:
            st.subheader("View Records")

            view_option = st.radio(
                "Choose records to display",
                ["All Patients", "Filter by Patient ID"],
                horizontal=True,
                key="all_patients_view"
            )

            if view_option == "Filter by Patient ID":
                available_codes = sorted(
                    all_visits["patient_code"].dropna().astype(str).unique().tolist()
                )

                selected_code = st.selectbox(
                    "Select Patient ID",
                    options=available_codes,
                    key="all_patients_filter"
                )

                display_all_df = all_visits[
                    all_visits["patient_code"].astype(str) == selected_code
                ].copy()
            else:
                display_all_df = all_visits.copy()

            st.dataframe(
                display_all_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = display_all_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Records as CSV",
                data=csv_data,
                file_name="patient_records.csv",
                mime="text/csv",
                key="download_patient_records"
            )

    except Exception as exc:
        st.error(f"Could not load all patient records: {exc}")
else:
    st.error("The database is unavailable, so all patient records cannot be displayed.")

