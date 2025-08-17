from keras_facenet import FaceNet
import cv2 as cv
from mtcnn import MTCNN
import numpy as np
import os
from numpy.linalg import norm

# Initialize
detector=MTCNN()
embedder=FaceNet()
known_embeddings={}

Faces="Faces"

# Take embedding of each img(first convert to rgb) and store in known embedding
for person_name in os.listdir(Faces):
    person_folder = os.path.join(Faces, person_name)
    
    for filename in os.listdir(person_folder):
        img=cv.imread(os.path.join(person_folder,filename))
        
        face=detector.detect_faces(img)
        
        if not face:
            print(f"No face detected in {os.path.join(person_folder,filename)}")
            continue
        
        x,y,w,h=face[0]['box']
        face_crop=img[y:y+h,x:x+w]
        
        img_rgb=cv.cvtColor(face_crop,cv.COLOR_BGR2RGB)
        img_rgb = cv.resize(img_rgb, (160, 160))
        
        embedding=embedder.embeddings([img_rgb])[0]
        
        if person_name not in known_embeddings:
            known_embeddings[person_name] = []
            
        known_embeddings[person_name].append(embedding)
    
print("Known faces loaded:",list(known_embeddings.keys()))

# Function to identify face

def identify_face(face_embedding, known_embeddings, threshold=1.0):
    min_dist = float('inf')
    identity = "Unknown"
    for name, db_emb_list in known_embeddings.items():
        for db_emb in db_emb_list:
            dist = norm(face_embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name
    if min_dist > threshold:
        identity = "Unknown"
    return identity

# Open camera 

vid=cv.VideoCapture(0)

while True:
    ret,frame=vid.read()
    
    if not ret:
        break
    
    faces=detector.detect_faces(frame)
    
    for face in faces:
        x,y,w,h=face['box']
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        face_crop=frame[y:y+h,x:x+w]
        face_crop_rgb=cv.cvtColor(face_crop,cv.COLOR_BGR2RGB)
        
        face_crop_rgb = cv.resize(face_crop_rgb, (160, 160))
        
        embedding=embedder.embeddings([face_crop_rgb])[0]
        
        name=identify_face(embedding,known_embeddings)
        
        cv.putText(frame,name,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        
    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
