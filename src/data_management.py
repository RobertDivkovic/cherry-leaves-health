# data_management.py
import os
import joblib
from tensorflow.keras.models import load_model

# Define a default version folder for consistency
VERSION = "03_modelling_and_evaluating"
BASE_OUTPUT_PATH = os.path.join("outputs", VERSION)

def load_model_and_metadata(model_path=None, image_shape_path=None, class_index_path=None):
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(BASE_OUTPUT_PATH, "cherry_leaf_mildew_model.h5")
    if image_shape_path is None:
        image_shape_path = os.path.join(BASE_OUTPUT_PATH, "image_shape.pkl")
    if class_index_path is None:
        class_index_path = os.path.join(BASE_OUTPUT_PATH, "class_indices.pkl")

    # Load all metadata and model
    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_index_path)

    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}