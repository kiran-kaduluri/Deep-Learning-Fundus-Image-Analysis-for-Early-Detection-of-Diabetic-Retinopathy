import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from flask import Flask, request, flash, render_template
from pymongo import MongoClient
import certifi

# ================= LOAD MODEL =================
model = load_model(r"model\inception-diabetic.h5")

# ================= FLASK =================
app = Flask(__name__)
app.secret_key = "abc"

# ================= MONGODB CONNECTION =================
MONGO_URI = "mongodb+srv://kiran:Kiran123@cluster0.wah3meq.mongodb.net/?appName=Cluster0"

client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=20000
)

db = client["retina_db"]
users_collection = db["users"]

# ================= HOME =================
@app.route('/')
def index():
    return render_template('index.html', pred="Login", vis="visible")

@app.route('/index')
def home():
    return render_template("index.html", pred="Login", vis="visible")

# ================= REGISTER =================
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        mail = request.form.get("emailid")
        mobile = request.form.get("num")
        pswd = request.form.get("pass")

        data = {
            'name': name,
            'mail': mail,
            'mobile': mobile,
            'psw': pswd
        }

        try:
            existing_user = users_collection.find_one({"mail": mail})
        except Exception:
            return render_template("register.html", pred="Database unavailable. Try again later.")

        if existing_user is None:
            users_collection.insert_one(data)
            return render_template("register.html", pred="Registration Successful, please login")
        else:
            return render_template("register.html", pred="You are already a member, please login")

    return render_template('register.html')

# ================= LOGIN =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "GET":
        user = request.args.get('mail')
        passw = request.args.get('pass')

        try:
            found_user = users_collection.find_one({"mail": user})
        except Exception:
            return render_template("login.html", pred="Database unavailable. Try again later.")

        if found_user and passw == found_user['psw']:
            flash("Logged in as " + str(user))
            return render_template('index.html', pred="Logged in as " + str(user), vis="hidden", vis2="visible")
        else:
            return render_template('login.html', pred="Invalid email or password")

    return render_template('login.html')

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    return render_template('logout.html')

# ================= PREDICTION =================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # file exists?
        if 'file' not in request.files:
            return "No file uploaded"

        f = request.files['file']

        # filename exists?
        if not f or f.filename.strip() == "":
            return "Please choose an image before predicting."

        basepath = os.path.dirname(__file__)

        upload_dir = os.path.join(basepath, "User_Images")
        os.makedirs(upload_dir, exist_ok=True)

        filepath = os.path.join(upload_dir, f.filename)

        # save file
        f.save(filepath)

        # ===== MODEL PREDICTION =====
        img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        prediction = np.argmax(model.predict(img_data), axis=1)

        classes = [
            'No Diabetic Retinopathy',
            'Mild NPDR',
            'Moderate NPDR',
            'Severe NPDR',
            'Proliferative DR'
        ]

        result = classes[prediction[0]]

        return render_template("prediction.html", prediction=result)

    return render_template("prediction.html")

# ================= RUN =================
if __name__ == "__main__":
    app.debug = False
    app.run()
