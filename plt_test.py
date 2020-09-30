import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")
print (f'matplotlib backend is {plt.get_backend()}')
# assert (plt.get_backend() == 'TkAgg') # If other 

img = np.zeros(shape=(512,512,3), dtype=np.uint8)
# plt.plot([1,2,3],[5,7,4])
# plt.figure(figsize=(24,32))

plt.imshow(img,  cmap='gray')
plt.show()

cv2.imshow('er', img)
cv2.waitKey()             
