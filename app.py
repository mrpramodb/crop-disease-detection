from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure static folder exists for storing uploaded images
UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "crop_disease_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ ERROR: Failed to Load Model! {e}")
    model = None  # Set model to None if loading fails

# Define disease information for all 15 classes
disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'description': 'A bacterial infection causing dark spots on pepper leaves.',
        'symptoms': 'Yellow halo around spots, leaf wilting, reduced yield.',
        'prevention': 'Use copper-based fungicides, ensure proper air circulation, and avoid overhead watering.',
        'image': 'prevention/pepper_bacterial_spot.jpg'
    },
    'Pepper__bell___healthy': {
        'description': 'A healthy pepper plant with no disease symptoms.',
        'symptoms': 'None, plant is in good condition.',
        'prevention': 'Maintain proper watering and use fertilizers as needed.',
        'image': 'prevention/healthy.jpg'
    },
    'Potato___Early_blight': {
        'description': 'A fungal disease that affects potato foliage and tubers.',
        'symptoms': 'Brown spots with concentric rings on leaves.',
        'prevention': 'Use resistant varieties, rotate crops, and apply fungicides as needed.',
        'image': 'prevention/potato_early_blight.jpg'
    },
    'Potato___Late_blight': {
        'description': 'A serious fungal disease that spreads fast in potatoes.',
        'symptoms': 'Dark, water-soaked lesions on leaves and stems.',
        'prevention': 'Remove infected plants, use fungicides, and avoid moisture.',
        'image': 'prevention/potato_late_blight.jpg'
    },
    'Potato___healthy': {
        'description': 'A healthy potato plant with no disease symptoms.',
        'symptoms': 'None, plant is in good condition.',
        'prevention': 'Maintain good field hygiene and ensure proper drainage.',
        'image': 'prevention/healthy.jpg'
    },
    'Tomato_Bacterial_spot': {
        'description': 'A bacterial infection causing black spots on tomato leaves and fruits.',
        'symptoms': 'Water-soaked spots turning brown or black.',
        'prevention': 'Use disease-free seeds, avoid overhead watering, and apply copper sprays.',
        'image': 'prevention/tomato_bacterial_spot.jpg'
    },
    'Tomato_Early_blight': {
        'description': 'A fungal disease causing dark spots on tomato leaves.',
        'symptoms': 'Brown spots with yellow edges on lower leaves.',
        'prevention': 'Use copper fungicides and avoid wet leaves overnight.',
        'image': 'prevention/tomato_early_blight.jpg'
    },
    'Tomato_Late_blight': {
        'description': 'A serious fungal disease that spreads fast in tomatoes.',
        'symptoms': 'Dark, water-soaked lesions on leaves and stems.',
        'prevention': 'Remove infected plants, use fungicides, and avoid moisture.',
        'image': 'prevention/tomato_late_blight.jpg'
    },
    'Tomato_Leaf_Mold': {
        'description': 'A fungal disease causing yellowing and moldy leaves in tomatoes.',
        'symptoms': 'Yellow spots on upper leaves, fuzzy mold on lower leaves.',
        'prevention': 'Improve ventilation, reduce humidity, and use fungicides.',
        'image': 'prevention/tomato_leaf_mold.jpg'
    },
    'Tomato_Septoria_leaf_spot': {
        'description': 'A common fungal disease that weakens tomato plants.',
        'symptoms': 'Small dark spots with gray centers on leaves.',
        'prevention': 'Remove infected leaves, use mulch, and avoid overhead watering.',
        'image': 'prevention/tomato_septoria_leaf_spot.jpg'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'description': 'Tiny mites that suck juice from tomato leaves.',
        'symptoms': 'Yellowing leaves, fine webbing on plant.',
        'prevention': 'Spray water to remove mites, use neem oil or insecticidal soap.',
        'image': 'prevention/tomato_spider_mite.jpg'
    },
    'Tomato__Target_Spot': {
        'description': 'A fungal disease causing brown rings on tomato leaves.',
        'symptoms': 'Circular brown lesions on leaves and stems.',
        'prevention': 'Use fungicides, rotate crops, and keep plants dry.',
        'image': 'prevention/tomato_target_spot.jpg'
    },
    'Tomato__Tomato_mosaic_virus': {
        'description': 'A virus causing distorted and mottled tomato leaves.',
        'symptoms': 'Mosaic patterns on leaves, reduced growth.',
        'prevention': 'Use virus-free seeds, control aphids, and remove infected plants.',
        'image': 'prevention/tomato_mosaic_virus.jpg'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'description': 'A virus spread by whiteflies affecting tomato plants.',
        'symptoms': 'Yellowing, curling leaves with stunted growth.',
        'prevention': 'Control whiteflies, use resistant varieties.',
        'image': 'prevention/tomato_yellowleaf_curl_virus.jpg'
    }
}

# Ensure class labels match `disease_info` keys
class_labels = list(disease_info.keys())

# Function to preprocess images before sending to the model
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ ERROR: Failed to Read Image!")
        return None
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("🛠 Received POST Request")

        file = request.files.get("file")  # Get file safely
        if not file or file.filename == "":
            return render_template("index.html", prediction="❌ No file selected!", image_path=None, disease_info={})

        # Save uploaded file
        filename = file.filename.replace(" ", "_")  
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess image
        img = preprocess_image(file_path)
        if img is None:
            return render_template("index.html", prediction="❌ Invalid image file!", image_path=None, disease_info={})

        # Make prediction
        prediction = model.predict(img)

        # Get predicted class
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]

        return render_template(
            "index.html",
            prediction=predicted_class,
            image_path=filename,
            disease_info=disease_info.get(predicted_class, {
                'description': 'No information available for this disease.',
                'symptoms': 'Unknown symptoms.',
                'prevention': 'No prevention tips available.',
                'image': None
            })
        )

    return render_template("index.html", prediction=None, image_path=None, disease_info={})

if __name__ == "__main__":
    app.run(debug=True)
