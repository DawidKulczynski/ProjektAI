import os
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import layers, models
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import dlib
import glob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

detector = dlib.get_frontal_face_detector()

DATASET_PATH = "UTKFace/"
IMG_SIZE = (200, 200)

def load_data():
    images = []
    ages = []
    for part in ["part1", "part2"]:
        files = glob.glob(os.path.join(DATASET_PATH, part, "*.jpg"))
        for file in files:
            filename = os.path.basename(file)
            age = int(filename.split("_")[0])
            img = cv2.imread(file)
            
            if img is None:
                print(f"Nie można wczytać obrazu: {file}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(img)
            if len(faces) > 0:
                x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
                img = img[y:y+h, x:x+w]
            else:
                print(f"Nie wykryto twarzy: {file}")
                continue
            
            try:
                img = cv2.resize(img, IMG_SIZE)
            except Exception as e:
                print(f"Błąd podczas skalowania obrazu: {file}, {e}")
                continue
            
            img = img / 255.0
            images.append(img)
            ages.append(age)
    return np.array(images), np.array(ages)

def load_validation_data():
    images = []
    ages = []
    files = glob.glob(os.path.join(DATASET_PATH, "part3", "*.jpg"))
    for file in files:
        filename = os.path.basename(file)
        age = int(filename.split("_")[0])
        img = cv2.imread(file)
        
        if img is None:
            print(f"Nie można wczytać obrazu: {file}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(img)
        if len(faces) > 0:
            x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
            img = img[y:y+h, x:x+w]
        else:
            print(f"Nie wykryto twarzy: {file}")
            continue
        
        try:
            img = cv2.resize(img, IMG_SIZE)
        except Exception as e:
            print(f"Błąd podczas skalowania obrazu: {file}, {e}")
            continue
        
        img = img / 255.0
        images.append(img)
        ages.append(age)
    return np.array(images), np.array(ages)

def train_model():
    X_train, y_train = load_data()
    X_val, y_val = load_validation_data()
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Predykcja wieku
    ])
    
    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mae'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    model.save('age_prediction_model.h5')
    return model

if not os.path.exists('age_prediction_model.h5'):
    model = train_model()
else:
    model = keras.models.load_model('age_prediction_model.h5', compile=False)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img)
    if len(faces) > 0:
        x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
        img = img[y:y+h, x:x+w]
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('index.html', error='Brak pliku w żądaniu')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Nie wybrano pliku')
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    img = preprocess_image(file_path)
    if img is None:
        return render_template('index.html', error='Nie udało się odczytać obrazu')
    
    predicted_age = model.predict(img)[0][0]
    predicted_age = int(predicted_age)
    
    if predicted_age < 10:
        age_category = "Niepełnoletni"
    elif predicted_age < 18:
        age_category = "Wymagana weryfikacja z obsługą"
    else:
        age_category = "Pełnoletni"
    
    return render_template('index.html', predicted_age=predicted_age, age_category=age_category)

if __name__ == '__main__':
    app.run(debug=True)
