# app/main.py
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from flask import Flask, jsonify, request, send_file
from app.config import UPLOAD_FOLDER, MODEL_PATH, PATCHED_FOLDER, PREDICTED_PATCH_FOLDER, PREDICTED_FOLDER
from app.image_upload.upload_handler import ImageUploadHandler
from app.image_processing.image_processor import ImageProcessor
from keras.models import load_model
from keras.optimizers import Adam

app = Flask(__name__)

handler = ImageUploadHandler()

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
        return jsonify(predicted_mask_url=predicted_mask_path), 200
    elif image_type == 'patch_image':
        predicted_mask_paths = handler.process_patch_image_prediction(model, PATCHED_FOLDER, filename,
                                                                      PREDICTED_PATCH_FOLDER)
        return jsonify(predicted_mask_urls=predicted_mask_paths), 200
    else:
        return jsonify(error="Invalid image type. Supported types: 'resized_image' or 'patch_image'"), 400


if __name__ == '__main__':
    app.run(debug=True)