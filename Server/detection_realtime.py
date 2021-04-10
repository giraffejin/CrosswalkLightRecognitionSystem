#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
import cv2
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from socket import *

sys.path.append("..")
CWD_PATH=os.getcwd()

# Load trained model
PATH_TO_FROZEN_GRAPH = os.path.join('tf_graph_rcnn', 'frozen_inference_graph_rcnn40k.pb')

PATH_TO_LABELS = os.path.join('legacy/training', 'labelmap.pbtxt')

PATH_TO_VIDEO="http://192.168.0.10:8091/?action=stream"

NUM_CLASSES=1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Rotate Video
def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)

    else:
        dst = null

    return dst


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

  sess=tf.Session(graph=detection_graph)

#find if there is red inside the cropped image(traffic light)
def detect_red(img, Threshold = 0.01):
  desired_dim = (30, 90)

  img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
    
  img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

  lower_red = np.array([0, 70, 50])
  upper_red = np.array([10, 255, 255])
  mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

  lower_red = np.array([170, 70, 50])
  upper_red = np.array([180, 255, 255])
  mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

  mask = mask0 + mask1

  global rate1
  rate1 = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

  if rate1 > Threshold:
    return True

  else:
    return False



#find if there is green inside the cropped image(traffic light)
def detect_green(img, Threshold = 0.01):
  desired_dim = (30, 90)
  img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
  img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  
  lower_green = np.array([25, 70, 50])
  upper_green = np.array([102, 255, 255])
  mask = cv2.inRange(img_hsv, lower_green, upper_green)

  global rate2
  rate2 = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])
  
  if rate2 > Threshold:
    return True

  else:
    return False


# Discriminate traffic light's signal
def read_traffic_lights(image, boxes, scores, classes):

  red_flag = False
  crop_img = image
  crop_img=crop_traffic_lights(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))


  if crop_img is not None:
      if detect_red(crop_img):
          red_flag = False
      else:
          red_flag=True

      if detect_green(crop_img):
          green_flag = True
      else:
          green_flag= False

      result_flag = red_flag & green_flag

  else:
      result_flag = False

  return result_flag

  
# Crop traffic light from the frame
def crop_traffic_lights(image, boxes, scores, classes, max_boxes_to_draw=20, min_score_thresh=0.5, traffic_light_label=1):
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[0] > min_score_thresh and classes[0] == traffic_light_label:
            ymin, xmin, ymax, xmax = tuple(boxes[0].tolist())
            (left, right, top, bottom) = (int(xmin * 500), int(xmax * 500), int(ymin * 500), int(ymax * 500))
            crop_img = image[top:bottom, left:right]
            crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
        else:
            crop_img = None

    return crop_img

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

HOST='192.168.0.10'

c = socket(AF_INET, SOCK_STREAM)
c.connect((HOST,3000))
print('ok')

i=1
k=0
l =0
while(video.isOpened()):
    ret, frame = video.read()
    frame = Rotate(frame, 270)
    frame = cv2.resize(frame, dsize=(500,500), interpolation = cv2.INTER_AREA)
    frame_expanded = np.expand_dims(frame, axis=0)

    if(i%6==1):
      k+=1
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: frame_expanded})

      crop_img=crop_traffic_lights(frame, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
      result_flag = read_traffic_lights(frame, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
      
      if result_flag:
        j =1
        print('go')  # okay to walk
      else:
        j =0
        print('stop') # unavailable to walk

      vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.50)

      cv2.imshow('Object detector', frame)
      l = l+j

      # Discriminate traffic light signal per second
      if(k%5==0):
        if(crop_img is None):
          print("none")
          flag = '1'
          c.send(flag.encode())
        elif(l%5==2):
          print("blink")
          flag = '2'
          c.send(flag.encode())
        elif(l==0):
          print("red")
          flag = '3'
          c.send(flag.encode())
        elif(l==5):
          print("green")
          flag = '4'
          c.send(flag.encode())
        l=0

    i+=1
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
