import cv2
import numpy as np
from matplatlib import pyplot as plt

img.gray = rgb2gray(img)
img.height, img.width = size(img)
feat = zeros(1,256)
class LBP:
     
    def get_pixel(img, center, x, y):
     value = 0
     try:
       if img[x][y]>=center:
       value[iter]
       except:
          pass
          return value
     def __init(self,img):
        self.image = cv2.imread(img, 0)
        self.img_rgb2gray = cv2.imread(img, 0)
        
        self.height = len(self.img)
        self.width = len(self.img[0])
        
     def execute(self):
       
       img.lbp = np.zeros(self.height, self.width, 3), np.uint8)
           
           for row in self.height:
              for column in self.width:
                 img[row, column] = self.LBP(row, img.column, 3)
              
         self.histogram(self.img, self.img_rgb2gray)
         self.display(self.img_rgb2gray)
         
         
     def _display images(self, img_rgb2gray, title):
         plt.figure()
         plt.axis("off")
         plt.imshow(img_rgb2gray, cmap='gray')
         plt.show()
         
    def histogram(self, pixel, row, column, default=0):
        try:
          return get_pixel[row, column]
          
          except IndexError:
               return default
     
     
    def _thresholded(self, center, neighbours):
    
    result = []
       for neighbour in neighbours:
           if neighbour >= center:
              result.append(1)
            else:
              result.append(0)
              return result
          
         
         
    
         




  for i=2 in height:
     for j=2 in width:
        height[i] = height[i] -1
        width[j] = width[j] -1
        
        neighbours = img.gray(i-1:i+1,j-1:j+1);
        bits = double(neighbours(:))
        threshold = bits(5);
        
       bits(5) = []
       bits = bits - threshold;
       bits = sign(bits)
       bits(bits < 0) = 0
       
       byte = sum(bits.*2.^(length(bits)-1 :-1 : 0)')
       feat(byte + 1 ) = feat(byte+1)+1;
        
        
        
        
        
        
        
