import cv2
import numpy as np
import tensorflow as tf
import os
from img2vi import *
from utils import *
def bbox_score(image):
	PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(graph=detection_graph, config=config) as sess:


		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_np_expanded = np.expand_dims(image_np, axis=0)

		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		scores = detection_graph.get_tensor_by_name('detection_scores:0')

		classes = detection_graph.get_tensor_by_name('detection_classes:0')

		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		(boxes, scores, classes, num_detections) = sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})


	return boxes,scores


def bbox_score_mini(image,detection_graph,sess):
	# PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
	# detection_graph = tf.Graph()
	# with detection_graph.as_default():
	# 	od_graph_def = tf.GraphDef()
	# 	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	# 		serialized_graph = fid.read()
	# 		od_graph_def.ParseFromString(serialized_graph)
	# 		tf.import_graph_def(od_graph_def, name='')
	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True

	# with tf.Session(graph=detection_graph, config=config) as sess:
		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_np_expanded = np.expand_dims(image_np, axis=0)

		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		scores = detection_graph.get_tensor_by_name('detection_scores:0')

		classes = detection_graph.get_tensor_by_name('detection_classes:0')

		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		(boxes, scores, classes, num_detections) = sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})

		return boxes, scores

def track_ssd(img_path="./data_try/0__0", t_type=0):
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

	tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
	tracker_type = tracker_types[t_type]
	global tracker

	# if int(minor_ver) < 3:
	if int(major_ver) < 3:
		tracker = cv2.Tracker_create(tracker_type)
	else:
		if tracker_type == 'BOOSTING':
			tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			tracker = cv2.TrackerGOTURN_create()
		if tracker_type == 'MOSSE':
			tracker = cv2.TrackerMOSSE_create()
		if tracker_type == "CSRT":
			tracker = cv2.TrackerCSRT_create()

	images_list = [img for img in os.listdir(img_path) if img.endswith(".jpg")]
	images_list = sort_img(images_list)
	# img_txt_path = img_path.replace(img_path.split("/")[0], "./txt")#?
	# txt_path = img_txt_path + "_txt"
	# txt_path = img_txt_path#?
	txt_path = img_path.replace('ksframes','bbox')

	frame = cv2.imread(os.path.join(img_path, images_list[0]))
	# Define an initial bounding box
	height, width, layers = frame.shape
	if not os.path.isdir(txt_path):
		os.makedirs(txt_path)
	PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(graph=detection_graph, config=config) as sess:
		for i in range(len(images_list)):
			frame = cv2.imread(os.path.join(img_path, images_list[i]))
			timer = cv2.getTickCount()
			bbox, score = bbox_score_mini(frame,detection_graph,sess)
			bbox = np.squeeze(bbox)[0]
			# fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			# score = np.squeeze(score)[0]
			bbox = bbox_transfer(bbox, height, width)
			write_txt(txt_path, images_list[i], bbox)
			# p1 = (int(bbox[0]), int(bbox[1]))
			# p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
			# cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

			# cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

			# Display FPS on frame
			# cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

			# # Display result
			# cv2.imshow("Tracking", frame)
			#
			# # Exit if ESC pressed
			# k = cv2.waitKey(1) & 0xff
			# if k == 27: break




if __name__ == "__main__":
	track_ssd()

