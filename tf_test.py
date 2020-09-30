import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} '3'supress everything

import tensorflow as tf
from misc.common import visualize_boxes_and_labels_on_image_array, category_index
# tf.keras.applications.MobileNetV2(
#     input_shape=None, alpha=1.0, include_top=True, weights='imagenet',
#     input_tensor=None, pooling=None, classes=1000, classifier_activation='softmax',
#     #**kwargs
# )

import cv2, numpy as np

pic_txt = (f'OpenCV version: {cv2.__version__}; TF version: {tf.__version__}')
print (pic_txt)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg") # otherwise it conflicts with opencv

# plt.plot([1,2,3],[5,7,4])
# plt.show()

# model_path = 'loaded_models/ssd_mobile_saved_model.pb'
# model_path = '/home/a/Projects/env/ssdMobileNetTf/loaded_models/ssd_mobile_saved_model.pb'
# model_path = '/home/a/Projects/env/ssdMobileNetTf/loaded_models/saved_model.pb'
# model_path = '/home/a/Projects/env/ssdMobileNetTf/loaded_models'
# model_path = 'loaded_models/'

src_pic_path = 'src_pictures/treet_car_ppl.jpg'
# src_pic_path = 'src_pictures/2.jpg'
model_path = 'loaded_models/ssd_mobilenet_v2_fpnlite_320x320_1'
model = tf.saved_model.load(model_path)

# trying to save 

loaded = tf.saved_model.load(model_path)
#print("MobileNet has {} trainable variables: {}, ...".format(
#          len(loaded.trainable_variables),
#          ", ".join([v.name for v in loaded.trainable_variables[:5]])))


# img = np.zeros(shape=(512,512,3), dtype=np.uint8)
img = cv2.imread(src_pic_path) 
to_show = img
img = img.reshape(
       (1, img.shape[0], img.shape[1], 3)).astype(np.uint8)

# img = load_image_into_numpy_array(src_pic_path)
results = model(img)

# different object detection models have additional results
# all of them are explained in the documentation
result = {key:value.numpy() for key,value in results.items()}
# print(result.keys())
if 1:
    for key in result:
        print('\n\n\n')
        print(key)
        print(result[key])


# print(results)
label_id_offset = 0 
visualize_boxes_and_labels_on_image_array(
      img[0],
      result['detection_boxes'][0],
      (result['detection_classes'][0] + label_id_offset).astype(int),
      result['detection_scores'][0],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.3,
      agnostic_mode=False,
      keypoints=None,
      keypoint_scores=None,
      keypoint_edges=None)

# cv2.imshow('img', to_show)

cv2.imshow(pic_txt, img[0])
cv2.waitKey()

# plt.figure(figsize=(24,32))
# plt.imshow(img[0])
# plt.show()
