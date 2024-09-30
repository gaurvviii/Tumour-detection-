import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model + higher level layers
model = MobileNetV2(weights='imagenet')

# Function to preprocess the input image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to the required size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image
    return img_array

# Function to predict tumor presence
def predict_tumor(img_path):
    img = prepare_image(img_path)  # Prepare the image
    preds = model.predict(img)  # Make prediction
    decoded_preds = decode_predictions(preds, top=3)[0]  # Decode predictions

    # Print top 3 predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i + 1}: {label} ({score:.2f})")

# Path to the input image
input_image_path = 'path/to/your/image.jpg'  # Update with your image path

# Run prediction
predict_tumor(input_image_path)
