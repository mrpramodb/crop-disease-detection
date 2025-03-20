🌾 Crop Disease Detection Web Application
This is a deep learning-based web application that detects crop diseases from leaf images. It is built using Flask as the backend, a pre-trained Keras model, and provides detailed information about the predicted disease.

📌 Features
Upload leaf images to detect diseases.
Provides disease description and prevention methods.
User-friendly web interface.
Powered by a trained deep learning model (crop_disease_model.h5).
Displays uploaded images and results dynamically.
🛠️ Technologies Used
Python 3
Flask
TensorFlow / Keras
OpenCV
HTML, CSS, JavaScript
Bootstrap (for styling)
📂 Folder Structure
bash
Copy
Edit
/myProject         # Your main Flask project
/static            # Folder to save uploaded images
/templates         # HTML templates (e.g., index.html)
/venv              # Python virtual environment
crop_disease_model.h5  # Trained model file
proj.py            # Optional additional scripts
README.md          # This file
🚀 How to Run
Clone the repo:

bash
Copy
Edit
git clone <your-repo-url>
cd your-project-folder
Activate virtual environment:

bash
Copy
Edit
.\venv\Scripts\activate  # For Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run Flask app:

bash
Copy
Edit
python myProject/app.py
Open in browser:

cpp
Copy
Edit
http://127.0.0.1:5000/
✨ Output Example
Image preview after upload.
Detected disease name.
Description and symptoms.
💡 Future Improvements
Add email notifications.
Improve model accuracy with more training data.
Deploy to cloud (Heroku/AWS/GCP).
