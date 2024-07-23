from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import statistics as stats
import pickle
import numpy as np
from threading import Lock 

app = Flask(__name__)

def midpoints(p1, p2):
    return (p1.x + p2.x)/2 * frame_shape[0], (p1.y + p2.y)/2 * frame_shape[1]

def distance(p1, p2):
    if isinstance(p1, tuple):
        x1, y1 = p1
        x2, y2 = p2
    else:
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_location(frame, landmark):
    ih, iw, _ = frame.shape
    x, y = int(landmark.x * iw), int(landmark.y * ih)
    return (x, y)

def display_landmarks(frame, landmarks):
    for landmark in landmarks:
        cv2.circle(frame, (landmark[0], landmark[1]), 2, (255, 0 ,255), cv2.FILLED)

def get_face(frame, center, height):
    ih, iw, _ = frame.shape

    if center[0] - height < 0:
        x0, x1 = 0, int(height*2)
    elif center[0] + height > iw:
        x0, x1 = int(iw - height*2), iw
    else:
        x0, x1 = int(center[0] - height), int(center[0] + height)

    if center[1] - height < 0:
        y0, y1 = 0, int(height*2)
    elif center[1] + height > ih:
        y0, y1 = int(ih - height*2), ih
    else:
        y0, y1 = int(center[1] - height), int(center[1] + height)
    
    crop = frame[y0:y1, x0:x1]
    resize = cv2.resize(crop,(48, 48))
    return resize

def find_distance(frame, landmarks):
    left = landmarks.landmark[133]
    right = landmarks.landmark[362]

    ih, iw, _ = frame.shape
    left = (int(left.x * iw), int(left.y * ih))
    right = (int(right.x * iw), int(right.y * ih))

    cv2.circle(frame, left, 2, (255, 0, 255), cv2.FILLED)
    cv2.circle(frame, right, 2, (255, 0, 255), cv2.FILLED)

    w = distance(left, right)
    W = 45
    f = 405
    d = (W * f) / w

    return d >= 50

def blinking_ratio(points, landmarks):
    left = (landmarks.landmark[points[0]].x * frame_shape[0], landmarks.landmark[points[0]].y * frame_shape[1])
    right = (landmarks.landmark[points[3]].x * frame_shape[0], landmarks.landmark[points[3]].y * frame_shape[1])
    top = midpoints(landmarks.landmark[points[1]], landmarks.landmark[points[2]])
    bottom = midpoints(landmarks.landmark[points[4]], landmarks.landmark[points[5]])
 
    hor_len = math.hypot(left[0]-right[0], left[1]-right[1])
    ver_len = math.hypot(top[0]-bottom[0], top[1]-bottom[1])

    return hor_len / (ver_len + 1e-7)

def find_blinking(landmarks):
    left_blink_ratio = blinking_ratio([33, 160, 158, 133, 153, 144], landmarks)
    right_blink_ratio = blinking_ratio([362, 385, 387, 263, 373, 380], landmarks)
    blink_ratio = round(stats.mean((left_blink_ratio, right_blink_ratio)), 2)

    if blink_ratio > 3:
        return True
    else:
        return False

def find_gazing(landmarks):
    left = distance(landmarks.landmark[5], landmarks.landmark[234])
    right = distance(landmarks.landmark[5], landmarks.landmark[454])

    threshold = 2.5
    result = "straight"
 
    if(left < right):
        ratio = right / left
        if(ratio > threshold):
            result = "left"
    elif(right < left):
        ratio = left / right
        if(ratio > threshold):
            result = "right"
    return result

def emotion_recognition(landmarks):
    center = get_location(frame, landmarks.landmark[195])
    forehead = get_location(frame, landmarks.landmark[10])
    chin = get_location(frame, landmarks.landmark[152])
    left_cheek = get_location(frame, landmarks.landmark[234])
    right_cheek = get_location(frame, landmarks.landmark[454])

    h = max(distance(center, forehead), distance(center, chin))
    w = max(distance(center, left_cheek), distance(center, right_cheek))
    
    crop = get_face(frame, center, h)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).copy()
    gray = np.reshape(gray, (1, 48, 48, 1))
    array = np.array(gray)

    prediction = classifier.predict(gray)
    return emotion_labels[prediction.argmax()]


cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
with open('model_for_comp .joblib', 'rb') as file:
    classifier = pickle.load(file)

emotion_duration = 0
old_emotion = None
distraction_duration = 0
state_lock = Lock()

def generate_frames():
    global emotion_duration, old_emotion, distraction_duration

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_shape = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                face_distance = find_distance(frame, landmarks)
                blinking = find_blinking(landmarks)
                gazing = find_gazing(landmarks) == 'straight'
                emotion = emotion_recognition(landmarks) in ['Happy', 'Neutral']

                with state_lock:  # Bảo vệ cập nhật trạng thái
                    if emotion != old_emotion:
                        emotion_duration = 0
                        old_emotion = emotion
                    else:
                        emotion_duration += 1

                    if not gazing or blinking:
                        distraction_duration += 1
                    else:
                        distraction_duration = 0

                    if emotion == False and emotion_duration >= 20:
                        print('Bạn cần nghỉ ngơi')
                    if distraction_duration >= 10:
                        print('Bạn đang bị phân tâm')

        print('------------------------------------------------------')
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
