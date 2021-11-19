import numpy as np
import cv2
from __future__ import print_function
try:
   from .imread import imread, imwrite, imread_multi, imread_from_blob, imwrite_multi
   from .imread import detect_format, supports_format
   from .imread import imload, imsave, imload_multi, imload_from_blob, imsave_multi
   from .imread_version import __version__
except ImportError as e:
     import destroyAllWindows
     import sys
     from scipy import misc, ndimage
     from matplotlib import pyplot as plt
     
     print('''\ '''.format(e), file=sys.stderr)
clc

path = ""
img = cv2.imread(path)

def Gaussian(img):
   img.Gaussian = img.misc()
   blurred_img = ndimage.Gaussian(faces, sigma = 3)
   ax[0].imshow(face)
   ax[0].set_title("Original Image")
   ax[0].set_x([])
   ax[0].set_y([])
   
   ax[1].imshow(blurred_img)
   ax[1].set_title("Blurred_Image")
   ax[1].set_x([])
   ax[1].set_y([])


def face_cascacde():
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x, y,w,h) in faces:
    img = cv2.rectangle(img,(x,y), (x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey_eh),(0,255,0),2)
      cv2.imshow('img'.img)
      cv2.waitkey(0)
      cv2.destroyAllWindows()
      

    


