# app/config.py
import os

# Configuration for file image_upload
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(os.getcwd(), 'models')
PATCHED_FOLDER = os.path.join(os.getcwd(), 'patched_images')
PREDICTED_FOLDER = os.path.join(os.getcwd(), 'output')
PREDICTED_PATCH_FOLDER = os.path.join(os.getcwd(), 'patch_output')
