import numpy as np
import imageio
import scipy.ndimage
import cv2

img = "tower.jpg"
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

def dodge(front,back):
    final_sketch = front*255/(255-back)
    final_sketch[final_sketch>255]=255
    final_sketch[back == 255] = 255
    
    return final_sketch.astype('uint8')

ss = imageio.imread(img)
gray = rgb2gray(ss)
i=255-gray

blur = scipy.ndimage.filters.gaussian_filter(i,sigma=13)
r = dodge(blur,gray)

cv2.imwrite('tower.jpg',r)

import cv2
import matplotlib.pyplot as plt
image = cv2.imread('tower.jpg')

gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(gray_img)
blur = cv2.GaussianBlur(invert,(21,21),0)
invertedblur = cv2.bitwise_not(blur)
sketch = cv2.divide(gray_img,invertedblur,scale = 256.0)
cv2.imwrite("sketch.jpg",sketch)
plt.imshow(sketch)

import cv2
import matplotlib.pyplot as plt
Sketch = cv2.imread("sketch.jpg")
plt.imshow(Sketch)