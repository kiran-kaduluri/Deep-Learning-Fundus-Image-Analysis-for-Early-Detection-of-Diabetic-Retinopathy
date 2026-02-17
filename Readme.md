# Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy

This project provides a Flask web app that uses a trained Inception-based deep learning model to classify fundus images into five diabetic retinopathy stages. It includes user registration/login, image upload, and prediction display, plus notebooks for training and experimentation.

## Features
- 5-class diabetic retinopathy classification
- Flask web UI for login, registration, and prediction
- MongoDB Atlas user storage
- Saved Keras model for quick inference
- Notebooks for model development and deployment

## Model
- Model file: model/inception-diabetic.h5
- Input size: 224x224 RGB
- Preprocessing: InceptionV3 preprocess_input
- Output classes:
	1. No Diabetic Retinopathy
	2. Mild NPDR
	3. Moderate NPDR
	4. Severe NPDR
	5. Proliferative DR

## Tech Stack
- Python, NumPy
- TensorFlow / Keras
- Flask
- MongoDB Atlas (pymongo + certifi)

## Project Structure
```
app.py
cloudant.ipynb
model/
	Diabetic_Retinopathy.ipynb
	Diabetic_Retinopathy_deployment_on_ibm_watson.ipynb
	Project.ipynb
	inception-diabetic.h5
static/
	css/
		styles.css
	images/
templates/
	index.html
	login.html
	logout.html
	prediction.html
	register.html
User_Images/
```

## Setup

### Prerequisites
- Python 3.8+ (recommended)
- A MongoDB Atlas cluster (or local MongoDB)

### Install dependencies
```
python -m venv .venv
.venv\Scripts\activate
pip install flask tensorflow keras pymongo certifi numpy
```

Optional (if PIL is missing):
```
pip install pillow
```

### Configure MongoDB and secrets
The app currently uses a hard-coded MongoDB connection string and Flask secret key. For security, replace them with environment variables.

Windows PowerShell example:
```
setx MONGO_URI "your-mongodb-uri"
setx FLASK_SECRET_KEY "your-secret-key"
```

Then update app.py to read from the environment:
```
MONGO_URI = os.getenv("MONGO_URI")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")
```

### Run the app
```
python app.py
```

Open http://127.0.0.1:5000

## Usage
1. Register a new user.
2. Log in.
3. Upload a fundus image on the prediction page.
4. View the predicted DR stage.

## Notebooks
- model/Diabetic_Retinopathy.ipynb: training experiments
- model/Project.ipynb: project workflow
- model/Diabetic_Retinopathy_deployment_on_ibm_watson.ipynb: deployment notes
- cloudant.ipynb: database experiments

## Data
The dataset is not included. Use publicly available fundus datasets (for example, EyePACS or APTOS) and follow the notebook guidance for preprocessing and training.

## Troubleshooting
- Missing model file: ensure model/inception-diabetic.h5 exists.
- MongoDB errors: verify connectivity and credentials.
- Upload issues: check write permissions for User_Images/.

## License
Add your license here.
