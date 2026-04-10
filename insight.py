import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_thresh=0.3)

dataset_path = './data/gallery/validation'
files = os.listdir(dataset_path)

features = {}
counter = 1
for file in files:
  print('prosessing image %d: %s'%(counter,file))

  img = cv2.imread(os.path.join(dataset_path, file))
  faces = app.get(img)
  features[file] = faces[0].normed_embedding

  counter+=1

print(features)

