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
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
for file in files:
  ext = os.path.splitext(file)[1].lower()
  if ext not in valid_exts:
    print('skipping unsupported file %d: %s' % (counter, file))
    counter += 1
    continue

  print('processing image %d: %s' % (counter, file))

  img = cv2.imread(os.path.join(dataset_path, file))
  if img is None:
    print('skipping unreadable image %d: %s' % (counter, file))
    counter += 1
    continue

  faces = app.get(img)
  if not faces:
    print('no face found in %d: %s' % (counter, file))
    counter += 1
    continue

  features.append(faces[0].normed_embedding)

  accepted_files.append(file)

  counter+=1

