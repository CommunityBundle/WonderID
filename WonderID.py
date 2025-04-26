import cv2
import cv2.data
import numpy as np
import os
import pickle
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_PROFILE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"
DB_PATH = "vector_database.pkl"
THRESHOLD = 0.85  

# --- INITIALIZE ---
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
face_profile_cascade = cv2.CascadeClassifier(HAAR_PROFILE_PATH)
# --- UTILITIES ---
def preprocess_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect frontal faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect profile faces in the image
    profile_faces = face_profile_cascade.detectMultiScale(gray, 1.3, 5)
    # Combine both frontal and profile faces
    faces = list(faces) + list(profile_faces)
    
    face_images = []
    bboxes = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w] 
        face = cv2.resize(face, (128, 128)) 
        face_images.append(face)
        bboxes.append((x, y, w, h))
    
    return face_images, bboxes

def extract_vector(face_img):
    if face_img.dtype != np.uint8:
        face_img = (face_img * 255).astype(face_recognition.uint8)
    
    if len(face_img.shape) == 2:  
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    encodings = face_recognition.face_encodings(rgb_face)
    if len(encodings) > 0:
        return encodings[0]
    return None

def save_database(db, path=DB_PATH):
    with open(path, "wb") as f:
        pickle.dump(db, f)

def load_database(path=DB_PATH):
    try:
        with open(path, 'rb') as f:
            vector_db = pickle.load(f)
    except FileNotFoundError:
        print("[INFO] No vector database found, starting fresh.")
        vector_db = {}
    return vector_db

def find_best_match(vector, db, threshold=THRESHOLD):
    if not db:
        return "Unknown", 0.0
    names, vectors = zip(*db.items())
    similarities = cosine_similarity([vector], vectors)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score < threshold:
        return "Unknown", best_score
    return names[best_idx], best_score

def display_database(db):
    if not db:
        print("[INFO] The database is empty.")
        return
    print("[INFO] Current Database:")
    for name, vector in db.items():
        print(f"Name: {name}, Vector: {vector}")

# --- MAIN LOOP ---
db = load_database()
mode = ''  # 
name_to_add = ""
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, bboxes = preprocess_faces(frame)
    for i, face in enumerate(faces):
        vector = extract_vector(face)
        if vector is None:
            continue 

        if mode == 'add':
            db[name_to_add] = vector
            save_database(db)
            cv2.putText(frame, f"Added: {name_to_add}", (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (bboxes[i][0], bboxes[i][1]), (bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3]), (0, 255, 0), 2)
        elif mode == 'detect':
            name, score = find_best_match(vector, db)
            label = f"{name} ({score:.2f})"
            x, y, w, h = bboxes[i]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('l'):  # Display database
        display_database(db)
    elif key == ord('a'):  # Add a new face
        mode = 'add'
        name_to_add = input("Enter name to add: ")
    elif key == ord('d'):  # Switch to detection mode
        mode = 'detect'
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
