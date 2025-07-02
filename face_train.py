import cv2 as cv
import numpy as np
from PIL import Image
import io


haarCascade = cv.CascadeClassifier("HaarCascades/frontal_face.xml")

def faces_extraction(people_data,minNeighbor):
    features = []
    labels = []
    label_map = {}
    label_idx = 0
    total_images = 0
    faces_found = 0

    for person in people_data:
        name = person['name']
        total_images += len(person['images'])

        if name not in label_map:
            label_map[name] = label_idx
            label_idx += 1
        
        for image in person['images']:
            image.seek(0)
            image_bytes = image.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("L")

            img = img.resize((500, int(500 * img.height / img.width)))
            img_np = np.array(img, dtype=np.uint8)

            faces_rect = haarCascade.detectMultiScale(img_np,1.1,minNeighbor)
            faces_found += len(faces_rect)

            for rectangle in faces_rect:
                (x,y,w,h) = rectangle
                cropped = img_np[y:y+h,x:x+w]
                features.append(cropped)
                labels.append(label_map[name])
    
    return features,labels,label_map,total_images,faces_found
            
