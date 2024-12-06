import logging
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    autoencoder_model_path = os.path.join(current_dir, "autoencoder_model.h5")
    autoencoder = load_model(autoencoder_model_path)
    logging.info("Anomaly detection model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load anomaly detection model: {e}")
    raise RuntimeError(f"Anomaly detection model could not be loaded: {e}")

try:
    tflite_model_path = os.path.join(current_dir, "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load TFLite model: {e}")
    raise RuntimeError(f"TFLite model could not be loaded: {e}")

def preprocess_image(image_data, target_size=(128, 128)):
    try:
        # If image_data is a file-like object, convert it to a numpy array
        if isinstance(image_data, BytesIO):
            image_data.seek(0)  # Ensure we're at the start of the file
            file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            # Assume it's a file path
            image = cv2.imread(image_data)

        if image is None:
            raise ValueError("Could not load the image. Ensure it's a valid image file.")
        
        # Resize with padding
        old_size = image.shape[:2]
        ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
        resized_image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
        delta_w = target_size[1] - new_size[1]
        delta_h = target_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        new_image = new_image.astype('float32') / 255.0  # Normalize to [0, 1]
        return np.expand_dims(new_image, axis=0)
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise ValueError(f"Image preprocessing failed: {e}")

def is_anomalous(image_path, threshold=0.02):
    try:
        processed_image = preprocess_image(image_path)
        reconstructed = autoencoder.predict(processed_image)
        reconstruction_error = np.mean((processed_image - reconstructed) ** 2)
        return reconstruction_error > threshold
    except Exception as e:
        logging.error(f"Anomaly detection failed: {e}")
        raise ValueError(f"Anomaly detection failed: {e}")

def predict_and_format_result(image_data):
    try:
        # Check for anomalies
        if is_anomalous(image_data):
            logging.info("Image detected as anomalous.")
            return "Anomalous", "N/A"

        # Preprocess for classification
        processed_image = preprocess_image(image_data)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Run inference
        interpreter.invoke()

        # Get classification result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
        result_class = class_names[predicted_class_index]
        max_probability = output_data[0][predicted_class_index]

        logging.info(f"Prediction: {result_class}, Probabilities: {output_data}")
        return result_class, f"{max_probability:.4f}"
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise ValueError(f"Prediction failed: {e}")
