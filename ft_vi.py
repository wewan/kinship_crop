import cv2
import sys
import tensorflow as tf
from face_detect import bbox_score
import numpy as np


def bbox_transfer(bbox, height, width):
	b0 = int(bbox[1] * width)
	b1 = int(bbox[0] * height)
	b2 = int((bbox[3] - bbox[1]) * width)
	b3 = int((bbox[2] - bbox[0]) * height)
	# bbox = (bbox[1] * width, bbox[0] * height, bbox[2] - bbox[0], bbox[3] - bbox[1])
	return (b0, b1, b2, b3)


def tracking_face(vi_path="try.avi", t_type=7):
	# Set up tracker.
	# Instead of MIL, you can also use
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

	tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
	tracker_type = tracker_types[t_type]

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

	# Read video
	video = cv2.VideoCapture(vi_path)

	# Exit if video not opened.
	if not video.isOpened():
		print("Could not open video")
		sys.exit()

	# Read first frame.
	ok, frame = video.read()
	if not ok:
		print('Cannot read video file')
		sys.exit()

	# Define an initial bounding box
	height, width, layers = frame.shape

	# box
	# bbox_transfer()
	bbox, score = bbox_score(frame)
	bbox = np.squeeze(bbox)[0]
	score = np.squeeze(score)[0]

	while True:
		if score > 0.7:
			break
		ok, frame = video.read()
		bbox, score = bbox_score(frame)
		bbox = np.squeeze(bbox)[0]
		score = np.squeeze(score)[0]

	bbox = bbox_transfer(bbox, height, width)#(269,47,62,80)
	# bbox = (bbox[1]*width,bbox[0]*height,bbox[2]-bbox[0],bbox[3]-bbox[1])
	# bbox = tuple(bbox)

	# bbox = (265, 48, 70, 82)# x,y, w, h
	# bbox = (48,151,128,186) # *H ymin, xmin, ymax, xmax
	# bbox = (85,270,228,332) # *W
	# bbox = (85,151,228,186) A -  151 85 (228-85) 186-151
	# bbox = (48,270,128,332) B


	# print(bbox.shape,score.shape)
	# Uncomment the line below to select a different bounding box
	# bbox = cv2.selectROI(frame, False)

	# Initialize tracker with first frame and bounding box
	ok = tracker.init(frame, bbox)
	# bbox = bbox_transfer(bbox)

	while True:
		# Read a new frame
		ok, frame = video.read()  # I think this line should be put after tracker.update
		if not ok:
			break

		# Start timer
		timer = cv2.getTickCount()

		# Update tracker
		ok, bbox = tracker.update(frame) # bbox --> [x1,y1,width,height]

		# Calculate Frames per second (FPS)
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

		# Draw bounding box
		if ok:
			# Tracking success
			p1 = (int(bbox[0]), int(bbox[1])) #(x1,y1)
			p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) #(x2,y2)
			cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
		else:
			# Tracking failure
			cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

		# Display tracker type on frame
		cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

		# Display FPS on frame
		cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

		# Display result
		cv2.imshow("Tracking", frame)

		# Exit if ESC pressed
		k = cv2.waitKey(1) & 0xff
		if k == 27: break


if __name__ == '__main__':
	tracking_face()


# # Set up tracker.
# # Instead of MIL, you can also use
# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#
# tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
# tracker_type = tracker_types[7]
#
# # if int(minor_ver) < 3:
# if int(major_ver) < 3:
#     tracker = cv2.Tracker_create(tracker_type)
# else:
#     if tracker_type == 'BOOSTING':
#         tracker = cv2.TrackerBoosting_create()
#     if tracker_type == 'MIL':
#         tracker = cv2.TrackerMIL_create()
#     if tracker_type == 'KCF':
#         tracker = cv2.TrackerKCF_create()
#     if tracker_type == 'TLD':
#         tracker = cv2.TrackerTLD_create()
#     if tracker_type == 'MEDIANFLOW':
#         tracker = cv2.TrackerMedianFlow_create()
#     if tracker_type == 'GOTURN':
#         tracker = cv2.TrackerGOTURN_create()
#     if tracker_type == 'MOSSE':
#         tracker = cv2.TrackerMOSSE_create()
#     if tracker_type == "CSRT":
#         tracker = cv2.TrackerCSRT_create()
#
# # Read video
# video = cv2.VideoCapture("try.avi")
#
# # Exit if video not opened.
# if not video.isOpened():
#     print("Could not open video")
#     sys.exit()
#
# # Read first frame.
# ok, frame = video.read()
# if not ok:
#     print('Cannot read video file')
#     sys.exit()
#
#
# # Define an initial bounding box
# height, width, layers = frame.shape
#
# # box
# # bbox_transfer()
# bbox, score = bbox_score(frame)
# bbox = np.squeeze(bbox)[0]
# score = np.squeeze(score)[0]
# bbox = bbox_transfer(bbox,height,width)
# # bbox = (bbox[1]*width,bbox[0]*height,bbox[2]-bbox[0],bbox[3]-bbox[1])
# # bbox = tuple(bbox)
#
# # bbox = (265, 48, 70, 82)# x,y, w, h
# # bbox = (48,151,128,186) # *H ymin, xmin, ymax, xmax
# # bbox = (85,270,228,332) # *W
# # bbox = (85,151,228,186) A -  151 85 (228-85) 186-151
# # bbox = (48,270,128,332) B
#
#
# # print(bbox.shape,score.shape)
# # Uncomment the line below to select a different bounding box
# # bbox = cv2.selectROI(frame, False)
#
# # Initialize tracker with first frame and bounding box
# ok = tracker.init(frame, bbox)
# # bbox = bbox_transfer(bbox)
#
# while True:
#     # Read a new frame
#     ok, frame = video.read() # I think this line should be put after tracker.update
#     if not ok:
#         break
#
#     # Start timer
#     timer = cv2.getTickCount()
#
#     # Update tracker
#     ok, bbox = tracker.update(frame)
#
#     # Calculate Frames per second (FPS)
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
#
#     # Draw bounding box
#     if ok:
#         # Tracking success
#         p1 = (int(bbox[0]), int(bbox[1]))
#         p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#         cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#     else :
#         # Tracking failure
#         cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
#
#     # Display tracker type on frame
#     cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
#
#     # Display FPS on frame
#     cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
#
#     # Display result
#     cv2.imshow("Tracking", frame)
#
#     # Exit if ESC pressed
#     k = cv2.waitKey(1) & 0xff
#     if k == 27 : break
