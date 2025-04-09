# data_management.py
import os
import joblib
import gdown
import zipfile
from tensorflow.keras.models import load_model

# Google Drive file ID of your model
GDRIVE_FILE_ID = "1GYpG0YDaNUGaxTx5c4CgVp5rbksmiptP"

# Google Drive file ID of the sample dataset zip
DATASET_ZIP_FILE_ID = "13_hajTD8dKprel4hu7GTllVBLaKf5lLD"

# Define versioned output folder
VERSION = "03_modelling_and_evaluating"
BASE_OUTPUT_PATH = os.path.join("outputs", VERSION)
MODEL_FILENAME = "cherry_leaf_mildew_model.h5"


def download_model_if_missing():
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    model_path = os.path.join(BASE_OUTPUT_PATH, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, model_path, quiet=False)

    return model_path

def download_sample_dataset():
    dataset_path = "inputs/heroku_sample_data/train"
    if os.path.exists(dataset_path):
        return  # Already downloaded

    print("Downloading sample dataset from Google Drive...")
    zip_url = f"https://drive.google.com/uc?id={DATASET_ZIP_FILE_ID}"
    zip_path = "inputs/cherry_sample.zip"

    os.makedirs("inputs", exist_ok=True)
    gdown.download(zip_url, zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("inputs/heroku_sample_data")
    os.remove(zip_path)


def load_model_and_metadata(model_path=None, image_shape_path=None, class_index_path=None):
    model_path = model_path or download_model_if_missing()
    image_shape_path = image_shape_path or os.path.join(BASE_OUTPUT_PATH, "image_shape.pkl")
    class_index_path = class_index_path or os.path.join(BASE_OUTPUT_PATH, "class_indices.pkl")

    model = load_model(model_path)
    image_shape = joblib.load(image_shape_path)
    class_indices = joblib.load(class_index_path)

    return model, image_shape, class_indices


def load_class_mapping(class_indices):
    return {v: k for k, v in class_indices.items()}