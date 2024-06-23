import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Parameters for image processing
RESIZE_DIM = (256, 256)
PATCH_SIZE = 128

class ImageProcessor:
    def __init__(self):
        self.labels_rgb = {
            "000": np.array([0, 0, 0]),
            "001": np.array([0, 0, 255]),
            "010": np.array([0, 255, 0]),
            "011": np.array([0, 255, 255]),
            "100": np.array([255, 0, 0]),
            "101": np.array([255, 255, 0]),
            "110": np.array([255, 0, 255]),
            "111": np.array([255, 255, 255])
        }

    def resize_image(self, image):
        return cv2.resize(image, RESIZE_DIM)

    def patchify(self, img):
        patches = []
        for i in range(0, img.shape[0], PATCH_SIZE):
            for j in range(0, img.shape[1], PATCH_SIZE):
                patch = img[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
                patches.append(patch)
        return patches

    def predict_resized_image(self, model, image_path):
        image = load_img(image_path, target_size=RESIZE_DIM)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        predicted_image = np.argmax(prediction, axis=-1)[0]
        rgb_image = self.label_to_rgb(predicted_image)
        return rgb_image

    def save_patches(self, patches, base_name, output_dir):
        saved_filenames = []
        for idx, patch in enumerate(patches):
            patch_filename = f"{base_name}_patch_{idx}.png"
            patch_filepath = os.path.join(output_dir, patch_filename)
            cv2.imwrite(patch_filepath, patch)
            saved_filenames.append(patch_filepath)
        return saved_filenames

    def predict_patch_images(self, model, patched_folder, filename, predicted_folder):
        predicted_patch_paths = []
        for i in range(4):
            patch_path = os.path.join(patched_folder, f"{filename}_patch_{i}.png")
            image = load_img(patch_path, target_size=(PATCH_SIZE, PATCH_SIZE))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # Predict the mask for the patch
            prediction = model.predict(image_array)
            predicted_patch = np.argmax(prediction, axis=-1)[0]
            rgb_patch = self.label_to_rgb(predicted_patch)

            # Save the predicted patch mask
            predicted_patch_path = os.path.join(predicted_folder, f"predicted_{filename}_patch_{i}.png")
            save_img(predicted_patch_path, rgb_patch)
            predicted_patch_paths.append(predicted_patch_path)

        return predicted_patch_paths

    def label_to_rgb(self, predicted_image):
        segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)

        for label, rgb_value in self.labels_rgb.items():
            segmented_img[(predicted_image == int(label, 2))] = rgb_value

        return segmented_img

    def save_predicted_mask(self, predicted_image, original_filename, output_folder):
        predicted_mask_path = os.path.join(output_folder, f"predicted_{os.path.basename(original_filename)}")
        save_img(predicted_mask_path, predicted_image)
        return predicted_mask_path
