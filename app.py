from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# === Load Models and Preprocessors ===
# Face Emotion Model
emotion_model = tf.keras.models.load_model('models/emotion_model.h5')

# Text Emotion Model
text_model = tf.keras.models.load_model('models/emotion_lstm_glove_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Max length used during training
MAX_LEN = 70

# === Emotion label to emoji path map ===
emoji_map = {
    'angry': 'emoji/angry.png',
    'fearful': 'emoji/fearful.png',
    'happy': 'emoji/happy.png',
    'love': 'emoji/happy.png',  # optional reuse
    'sad': 'emoji/sad.png',
    'surprised': 'emoji/surprised.png'
}

# For face emotion index prediction
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}
emoji_dist = {
    0: "emoji/angry.png", 1: "emoji/disgusted.png",
    2: "emoji/fearful.png", 3: "emoji/happy.png",
    4: "emoji/neutral.png", 5: "emoji/sad.png",
    6: "emoji/surprised.png"
}

# === Routes ===

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-face-emotion', methods=['POST'])
def predict_face_emotion():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    ).detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({'emotion': 'No face', 'emoji': 'emoji/neutral.png'})

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    face = cv2.resize(roi_gray, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    prediction = emotion_model.predict(face)[0]
    emotion_index = int(np.argmax(prediction))
    emotion = emotion_dict[emotion_index]
    emoji_path = emoji_dist[emotion_index]

    return jsonify({'emotion': emotion, 'emoji': emoji_path})

@app.route('/text-emotion', methods=['GET', 'POST'])
def text_emotion():
    if request.method == 'POST':
        user_text = request.form['text']

        sequence = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
        pred = text_model.predict(padded)[0]
        emotion_index = np.argmax(pred)
        emotion_label = label_encoder.inverse_transform([emotion_index])[0]
        emoji_path = emoji_map.get(emotion_label, 'emoji/neutral.png')

        return render_template('text_emotion.html',
                               emotion=emotion_label.capitalize(),
                               emoji=emoji_path,
                               user_text=user_text)
    return render_template('text_emotion.html')

@app.route('/video')
def video():
    return render_template('face_emotion.html')

# No need for /video_feed route since webcam is used via browser-side JS

if __name__ == '__main__':
    app.run(debug=True)
