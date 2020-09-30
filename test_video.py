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

pic_width = 800
pic_height = int(pic_width/1.2)
pic_height = 600
src_pic_path = 'src_pictures/treet_car_ppl.jpg'
# src_pic_path = 'src_pictures/2.jpg'
model_path = 'loaded_models/ssd_mobilenet_v2_fpnlite_320x320_1'
model_path = 'tf_save'
# model_path = 'loaded_models/saved_model.pb'
model = tf.saved_model.load(model_path)


# saved_model_path = "tf_save"
# tf.saved_model.save(model, saved_model_path)

# model.summary()

video_src = '/home/a/Videos/U524803_1_189_0.avi' # day
video_src = '/home/a/Videos/U524802_12.avi'
video_src = '/home/a/Videos/U524802_night_gain4_exp1689_2.avi' # night
video_src = 'http://136.169.226.9/001-999-171/index.m3u8?token=e2dc5e03e6f747d981c454626e5ea099'
video_src = 'http://136.169.226.52/1565342802/index.m3u8?token=2a569f42feca4e44ab21c29be6f183d2'
# video_src = '/home/a/Videos/U524802_day_gain1_exp116.avi'
# video_src = '/home/a/Videos/U524802_12.avi'
# video_src = '/home/a/Videos/U524802_11.avi'
# video_src = '/home/a/Videos/U524803_1.avi' # deep night bad focus

# video_src = 'rtsp://172.16.20.97/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'

cap = cv2.VideoCapture(video_src)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600) doesn't work
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print ('size - ', cap.get(3), cap.get(4))

while True:
    ret, img = cap.read()
    
    if not ret: # reload if the end of file
        cap = cv2.VideoCapture(video_src)
        ret, img = cap.read()

    # img = np.zeros(shape=(512,512,3), dtype=np.uint8)
    #img = cv2.imread(src_pic_path) 
    #to_show = img
    dim = (800, 600)
    img = cv2.resize(img, (pic_width, pic_height))
    # img = load_image_into_numpy_array(src_pic_path)
    
    img = img.reshape(
        (1, img.shape[0], img.shape[1], 3)).astype(np.uint8)
    
    results = model(img)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key:value.numpy() for key,value in results.items()}
    # print(result.keys())
    if 0:
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
        max_boxes_to_draw=15,
        min_score_thresh=.4,
        agnostic_mode=False,
        keypoints=None,
        keypoint_scores=None,
        keypoint_edges=None)

    # cv2.imshow('img', to_show)

    cv2.imshow(pic_txt, img[0])
    k = cv2.waitKey(1)
    if k ==27:
        break

cap.release()
cv2.destroyAllWindows()