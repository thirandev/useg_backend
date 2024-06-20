import os
import cv2
from flask import request
from werkzeug.utils import secure_filename
from app.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, PREDICTED_FOLDER
from app.image_processing.image_processor import ImageProcessor


class ImageUploadHandler:
    def __init__(self):
        self.processor = ImageProcessor()

    def process_upload(self, file):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read the uploaded image
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Error reading uploaded image")

        # Resize the image
        resized_image = self.processor.resize_image(image)
        resized_image_filename = f"{os.path.splitext(filename)[0]}_resized.png"
        resized_image_filepath = os.path.join(UPLOAD_FOLDER, resized_image_filename)
        cv2.imwrite(resized_image_filepath, resized_image)

        # Patchify the resized image
        image_patches = self.processor.patchify(resized_image)

        # Save image patches
        base_name = os.path.splitext(filename)[0]
        patched_image_dir = os.path.join(os.getcwd(), 'patched_images')
        os.makedirs(patched_image_dir, exist_ok=True)
        saved_patch_filenames = self.processor.save_patches(image_patches, base_name, patched_image_dir)

        # Prepare response with filenames and URLs of saved patches
        patched_image_urls = {}
        for idx, patch_filepath in enumerate(saved_patch_filenames):
            patched_image_urls[
                f"{base_name}_patch_{idx}"] = f"http://{request.host}/{os.path.relpath(patch_filepath, os.getcwd())}"

        response_data = {
            'message': 'Image uploaded, resized, and patched successfully.',
            'file_name': filename,
            'resizedImage': f"http://{request.host}/{os.path.relpath(resized_image_filepath, os.getcwd())}",
            'patched_images': patched_image_urls
        }

        return response_data

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def process_resized_image_prediction(self, model, image_path):
        predicted_mask = self.processor.predict_resized_image(model, image_path)
        predicted_mask_path = self.processor.save_predicted_mask(predicted_mask, image_path, PREDICTED_FOLDER)
        return predicted_mask_path

    def blend_patches(self, filename):
        patch_folder = os.path.join(os.getcwd(), 'patch_output')
        blend_folder = os.path.join(os.getcwd(), 'blended_output')
        os.makedirs(blend_folder, exist_ok=True)

        blended_image_path = self.processor.blend_patches(patch_folder, filename, blend_folder)
        return blended_image_path

    def process_patch_image_prediction(self, model, patched_folder, filename, predicted_folder):
        predicted_mask_paths = self.processor.predict_patch_images(model, patched_folder, filename, predicted_folder)
        return predicted_mask_paths

