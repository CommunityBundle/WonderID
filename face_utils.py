import face_recognition
import numpy as np
import cv2

def preprocess_frame(frame):
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return rgb

def encode_faces(frame):
    rgb = preprocess_frame(frame)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    return locations, encodings

def match_face(known_encs, known_names, encoding, tol=0.45):
    if len(known_encs) == 0:
        return "Unknown"
    dists = face_recognition.face_distance(known_encs, encoding)
    min_idx = np.argmin(dists)
    return known_names[min_idx] if dists[min_idx] < tol else "Unknown"
