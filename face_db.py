import pickle
import os

DB_PATH = "vector_db.pkl"

def load_vector_db():
    if not os.path.exists(DB_PATH):
        return [], []
    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)
        return data["encodings"], data["names"]

def save_vector_db(encodings, names):
    with open(DB_PATH, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
