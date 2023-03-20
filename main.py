from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
from scipy.spatial import procrustes
import pymongo

app = Flask(__name__)

client = pymongo.MongoClient("mongodb+srv://ethicalbyte:Ethicalbyte@cluster0.bxhz060.mongodb.net/ethicalbyte?retryWrites=true&w=majority")
db = client["jaipurhackathon"]
users_collection = db["users"]

def img_to_landmark(img_path):
    img = cv2.imread(img_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks = np.array([[landmark.x, landmark.y,landmark.z] for landmark in results.multi_face_landmarks[0].landmark])
    return landmarks

def image_to_landmark(img_path):
    # Initialize MediaPipe Face Mesh model
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Read input image
    image = cv2.imread(img_path)

    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get facial landmarks
    results = face_mesh.process(image_rgb)
    landmark_dict = {}
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        for i, landmark in enumerate(landmarks.landmark):
            landmark_dict[f"point_{i}"] = {"x": landmark.x, "y": landmark.y, "z": landmark.z}
    return landmark_dict

def compare_images_distance(landmarks1,landmarks2):
    points1 = []
    points2 = []
    for i in range(68):
        point1 = landmarks1[f"point_{i}"]
        point2 = landmarks2[f"point_{i}"]
        points1.append([point1["x"], point1["y"],point1["z"]])
        points2.append([point2["x"], point2["y"],point2["z"]])
    mean1 = np.mean(points1, axis=0)
    std1 = np.std(points1)
    points1_norm = (points1 - mean1) / std1
    mean2 = np.mean(points2, axis=0)
    std2 = np.std(points2)
    points2_norm = (points2 - mean2) / std2

    # Find the optimal rotation and translation
    mtx1, mtx2, disparity = procrustes(points1_norm, points2_norm)

    # Compute the Procrustes distance
    procrustes_distance = np.sum(np.square(mtx1 - mtx2))
    return procrustes_distance

def data_json(name,mobile_no,email,landmark):
    json_data = {
        "name": name,
        "mobile_no": mobile_no,
        "email": email,
        "landmark": landmark
    }
    
    users_collection.insert_one(json_data)

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    mobile_no = request.form.get('mobile_no')
    email = request.form.get('email')
    img_file = request.files['file']
    img_path = img_file.filename
    img_file.save(img_path)

    # Get facial landmarks from the input image
    landmarks = image_to_landmark(img_path)

    # Save user data and landmarks to MongoDB
    data_json(name, mobile_no, email, landmarks)

    return jsonify({"message": "User registered successfully."})


@app.route('/checkin', methods=['POST'])
def checkin():
    mobile_no = request.form.get('mobile_no')
    img_file = request.files['file']
    img_path =  img_file.filename
    img_file.save(img_path)

    # Retrieve user's data from MongoDB
    user = users_collection.find_one({"mobile_no": mobile_no})
    if user is None:
        return jsonify({"message": "User not found."}), 404

    # Extract landmarks from the uploaded image
    new_landmarks = image_to_landmark(img_path)

    # Compare the new landmarks with the stored landmarks
    stored_landmarks = user['landmark']
    distance = compare_images_distance(stored_landmarks, new_landmarks)

    # Check if the distance is below a certain threshold
    if distance <= 0.1:
        return jsonify({"message": "Check-in successful."}), 200
    else:
        return jsonify({"message": "Check-in failed. Please try again."}), 400

if __name__ == '__main__':
    app.run(debug=True)