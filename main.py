# app/main.py
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from app.config import UPLOAD_FOLDER, MODEL_PATH, PATCHED_FOLDER, PREDICTED_PATCH_FOLDER, PREDICTED_FOLDER
from app.image_upload.upload_handler import ImageUploadHandler
from keras.models import load_model
from keras.optimizers import Adam

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

handler = ImageUploadHandler()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PATCHED_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_PATCH_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)


# POST endpoint to image_upload and process an image
@app.route('/upload-process-image', methods=['POST'])
def upload_process_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    if file and handler.allowed_file(file.filename):
        try:
            response_data = handler.process_upload(file)
            return jsonify(response_data), 200
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except Exception as e:
            return jsonify(error="Internal server error"), 500
    else:
        return jsonify(error="Invalid file type"), 400


# Serve the uploaded and processed image patches
@app.route('/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_file(filename)


# GET endpoint to predict the mask for the given image
@app.route('/predict-mask', methods=['GET'])
def predict_mask():
    filename = request.args.get('filename')
    image_type = request.args.get('type')
    model_type = request.args.get('model_type')
    if not filename or not image_type:
        return jsonify(error="Filename and type parameters are required"), 400

    model_path = os.path.join(MODEL_PATH, model_type)
    model_path = f"{model_path}.hdf5"
    print(model_path)
    model = load_model(model_path, compile=False)
    metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=metrics)

    if image_type == 'resized_image':
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_path = f"{image_path}_resized.png"
        print(image_path)
        if not os.path.exists(image_path):
            return jsonify(error="Resized image not found"), 404
        predicted_mask_path = handler.process_resized_image_prediction(model, image_path)
        predicted_mask_url = f"http://{request.host}/{os.path.relpath(predicted_mask_path, os.getcwd())}"
        return jsonify(predicted_mask_url=predicted_mask_url), 200
    elif image_type == 'patch_image':
        predicted_mask_paths = handler.process_patch_image_prediction(model, PATCHED_FOLDER, filename,
                                                                      PREDICTED_PATCH_FOLDER)
        predicted_mask_urls = [f"http://{request.host}/{os.path.relpath(path, os.getcwd())}" for path in predicted_mask_paths]
        return jsonify(predicted_mask_urls=predicted_mask_urls), 200
    else:
        return jsonify(error="Invalid image type. Supported types: 'resized_image' or 'patch_image'"), 400


# GET endpoint to combine the patch masks into a single image using OpenCV seamless cloning
@app.route('/combine-patches', methods=['GET'])
def combine_patches():
    filename = request.args.get('filename')
    if not filename:
        return jsonify(error="Filename parameter is required"), 400

    try:
        patch_0_path = os.path.join(PREDICTED_PATCH_FOLDER, f"predicted_{filename}_patch_0.png")
        patch_1_path = os.path.join(PREDICTED_PATCH_FOLDER, f"predicted_{filename}_patch_1.png")
        patch_2_path = os.path.join(PREDICTED_PATCH_FOLDER, f"predicted_{filename}_patch_2.png")
        patch_3_path = os.path.join(PREDICTED_PATCH_FOLDER, f"predicted_{filename}_patch_3.png")

        patch_0 = cv2.imread(patch_0_path)
        patch_1 = cv2.imread(patch_1_path)
        patch_2 = cv2.imread(patch_2_path)
        patch_3 = cv2.imread(patch_3_path)

        combined_image = np.zeros((256, 256, 3), dtype=np.uint8)

        combined_image[0:128, 0:128, :] = patch_0
        combined_image[0:128, 128:256, :] = patch_1
        combined_image[128:256, 0:128, :] = patch_2
        combined_image[128:256, 128:256, :] = patch_3

        # Create masks for seamless cloning
        mask = np.ones((128, 128, 3), dtype=np.uint8) * 255

        center_0 = (64, 64)
        center_1 = (192, 64)
        center_2 = (64, 192)
        center_3 = (192, 192)

        combined_image = cv2.seamlessClone(patch_0, combined_image, mask, center_0, cv2.NORMAL_CLONE)
        combined_image = cv2.seamlessClone(patch_1, combined_image, mask, center_1, cv2.NORMAL_CLONE)
        combined_image = cv2.seamlessClone(patch_2, combined_image, mask, center_2, cv2.NORMAL_CLONE)
        combined_image = cv2.seamlessClone(patch_3, combined_image, mask, center_3, cv2.NORMAL_CLONE)

        combined_image_path = os.path.join(PREDICTED_FOLDER, f"{filename}_combined_seamlessClone.png")
        cv2.imwrite(combined_image_path, combined_image)

        return jsonify(message="Combined image saved successfully", combined_image_path=combined_image_path), 200
    except Exception as e:
        return jsonify(error="Error combining patches: " + str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
