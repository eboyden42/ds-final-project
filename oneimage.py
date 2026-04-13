import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_thresh=0.05)
data_path = './data/probe/training/obama_very_sad_2.jpg' 

def scale_for_viewing(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

print('checking image: %s' % (data_path))

img = cv2.imread(data_path)
if img is None:
    print(f"Error: Could not read image at {data_path}")
else:
    faces = app.get(img)

    for i, face in enumerate(faces):
        bbox = face.bbox
        x1, y1, x2, y2 = bbox.astype(int)

        print(f"Face {i} Bounding Box: Top-Left ({x1}, {y1}), Bottom-Right ({x2}, {y2})")
        print(f"Detection Score: {face.det_score:.4f}")
        print(f"Pose: {face.pose}")
        print(f"KPS: {face.kps}")
        print(f"calculated angle: {np.arctan((face.kps[0][1] - face.kps[1][1]) / (face.kps[0][0] - face.kps[1][0])) * 180 / np.pi:.2f} degrees")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kps in face.kps:
            cv2.circle(img, (int(kps[0]), int(kps[1])), 15, (0, 0, 255), -1)

    cv2.imshow('Detection Check', scale_for_viewing(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
