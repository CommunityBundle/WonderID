import cv2
from face_db import load_vector_db, save_vector_db
from face_utils import preprocess_frame, encode_faces, match_face

def main():
    encs, names = load_vector_db()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        locs, new_encs = encode_faces(frame)
        for loc, enc in zip(locs, new_encs):
            name = match_face(encs, names, enc)
            top, right, bottom, left = [i*4 for i in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("WonderID", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            name = input("Enter name: ").strip()
            _, new_enc = encode_faces(frame)
            if new_enc:
                encs.append(new_enc[0])
                names.append(name)
                save_vector_db(encs, names)
                print(f"[INFO] Added: {name}")
            else:
                print("[WARN] No face found.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
