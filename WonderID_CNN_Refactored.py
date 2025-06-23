import os
import cv2
import numpy as np
import torch
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from pathlib import Path
# ---MODEL INITIALIZATION---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# ---DATABASE SETUP---
DATASET_DIR = 'known_faces'
os.makedirs(DATASET_DIR, exist_ok=True)
known_faces = {}
# --- UTILITIES ---
def load_known_faces():
    print("loading known faces")
    known_faces.clear()
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue
        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = Image.open(img_path).convert('RGB')
            face_tensors = mtcnn(img)
            if face_tensors is not None:
                for face_tensor in face_tensors:
                    with torch.no_grad():
                        embedding = resnet(face_tensor.unsqueeze(0).to(device))
                        embeddings.append(embedding.squeeze(0).cpu())

        if embeddings:
            known_faces[person] = torch.stack(embeddings).mean(0)

def preprocessor(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mtcnn receive RGB as input
    boxes, confidence_score = mtcnn.detect(img)
    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)
        return faces, boxes
    else:
        #print("No faces found!")
        return None, None

def extract_feature_vectors(faces, boxes):
    embeddings = []
    if faces is not None:
        for face in faces:
            with torch.no_grad():
                # ResNet expects batches => we need to add a batch dimension through unsqueeze(0) => image becomes [1,3,160,160]
                # the image then got processed and resnet return a 1D vector with 512 feature values 
                # then move back to cpu for simple processing and squeeze(0) to remove the batch dimension 
                embedding = resnet(face.unsqueeze(0).to(device)).cpu().squeeze(0)
                embeddings.append(embedding)
        return embeddings, boxes
    else:
        #print("Unable to retrieve features of a face!")
        return None, None 

def cosine_similarity(t1, t2):
    return torch.nn.functional.cosine_similarity(t1, t2, dim=0).item()

def detected_features(embeddings, boxes):
    if embeddings is None or boxes is None:
        return None  
    results = []
    for embedding, box in zip(embeddings, boxes):
        best_match = "unknown"
        best_score = 0.7 

        for name, known_embedding in known_faces.items():
            score = cosine_similarity(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_match = name

        results.append((best_match,best_score, box)) 
    return results

# ---MAIN LOOP---     
load_known_faces()
recognizing = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if recognizing:
        start_time = time.time()
        # Load faces
        faces, boxes = preprocessor(frame)
        detect_time = time.time()

        if faces is not None:
            # Get feature_vectors
            embeddings, boxes = extract_feature_vectors(faces, boxes)
            # Retrieve features of a known face 
            results = detected_features(embeddings, boxes)
            recognize_time = time.time()
            if results:
                for name,score,box in results:
                    x1, y1, x2, y2 = map(int, box)
                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Display the name & similiarity scores of the recognized person
                    label = f"{name} ({score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Detect time: {detect_time - start_time:.3f}s, Recognize time: {recognize_time - detect_time:.3f}s")
        else:   
            boxes = None

    cv2.putText(frame, "'a': add face  'r': recognize  'q': quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow("face recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('a'):
        name = input("Enter name: ").strip()
        save_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        count = len(os.listdir(save_dir)) + 1
        img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        load_known_faces()

    elif key == ord('r'):
        recognizing = not recognizing
cap.release()
cv2.destroyAllWindows()
