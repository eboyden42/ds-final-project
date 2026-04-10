import torch
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_thresh=0.5)

dataset_path = './data/gallery/training'
files = os.listdir(dataset_path)

accepted_files = []
features = []
counter = 1
for file in files:
#   if not file.endswith('.jpg'):
#     continue
  print('prosessing image %d: %s'%(counter,file))

  img = cv2.imread(os.path.join(dataset_path, file))
  print
  faces = app.get(img)
  features.append(faces[0].normed_embedding)

  accepted_files.append(file)

  counter+=1

