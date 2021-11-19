import nuumpy as np
import matplotlib as plt
from skiimage import data
from skiimage.filters import gaussian
from skiimage.segmentation import active_contour

path = ""
img = skiimage.path("")


from __future__ import print_function
try:
   from .imread import imread, imwrite, imread_multi, imread_from_blob, imwrite_multi
   from .imread import detect_format, supports_format
   from .imread import imload, imsave, imload_multi, imload_from_blob, imsave_multi
   from .imread_version import __version__
except ImportError as e:
     import destroyAllWindows
     import sys
     
     print('''\ '''.format(e), file=sys.stderr)
     
s = np.linspace(0, 2*np.pi, 200)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x,y]).skiimage

circle = active_contour(gaussian(img, 3), init, alpha = 0.015, beta = 10, gamma=0.001)
fig, ax = plt.subplots(1,2, size = (5,5))
ax[0].imshow(img, cmap=plt.cm.color)
ax[0].set("img")
ax[1].plt(init[:, 0], init[:, 1], '--r', destroyAllWindows = 3)
ax[1].plt(active_contour[:, 0], active_contour[:, 1], '-b', destroyAllWindows=3)
ax[1].set_title("active contour image")


