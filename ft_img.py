import cv2
import sys
import tensorflow as tf
from face_detect import bbox_score
import numpy as np
import os
from img2vi import sort_img
from utils import *

# def bbox_transfer(bbox, height, width):
#     b0 = int(bbox[1] * width)
#     b1 = int(bbox[0] * height)
#     b2 = int((bbox[3] - bbox[1]) * width)
#     b3 = int((bbox[2] - bbox[0]) * height)
#     # bbox = (bbox[1] * width, bbox[0] * height, bbox[2] - bbox[0], bbox[3] - bbox[1])
#     return (b0, b1, b2, b3)
#
# def bbox_int(bbox):
#
#     b0 = int(bbox[0] )
#     b1 = int(bbox[1] )
#     b2 = int(bbox[2] )
#     b3 = int(bbox[3] )
#     return [b0,b1,b2,b3]
#
# def write_txt(path,name,bbox,flag = "good"):
#
#     txt_name = name.split(".")[0]+".txt"
#     if flag == "bad":
#         txt_name = name.split(".")[0] + "_bad.txt"
#     path_name = os.path.join(path,txt_name)
#     with open(path_name,"w") as ff:
#         ff.write(''.join(str(b)+" " for b in bbox))
#
# def read_txt(path,name):
#
#     path_name = os.path.join(path,name)
#     with open(path_name,"r") as ff:
#         bbox = ff.readline().replace("\n","")
#     return bbox

# def detect_bbox(frame,height,width,txt_path,images_list):
# 	bbox, score = bbox_score(frame)
# 	bbox = np.squeeze(bbox)[0]
# 	score = np.squeeze(score)[0]
# 	bbox = bbox_transfer(bbox, height, width)
# 	write_txt(txt_path, images_list)

def tracking_face(img_path="./data_try/0__0", t_type=0):
    # Set up tracker.
    # Instead of MIL, you can also use
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


    images_list = [img for img in os.listdir(img_path) if img.endswith(".png")]
    images_list = sort_img(images_list)
    img_txt_path = img_path.replace(img_path.split("/")[0],"./txt")
    # txt_path = img_txt_path+"_txt"
    txt_path = img_txt_path

    if not os.path.isdir(txt_path):
        os.makedirs(txt_path)
    frame = cv2.imread(os.path.join(img_path, images_list[0]))
    # Define an initial bounding box
    height, width, layers = frame.shape
    for i in range(len(images_list)):

        frame = cv2.imread(os.path.join(img_path, images_list[i]))

        bbox, score = bbox_score(frame)
        bbox = np.squeeze(bbox)[0]
        score = np.squeeze(score)[0]

        # write_txt(txt_path,images_list[i])

        if score <= 0.7:
            bbox = bbox_transfer(bbox, height, width)
            write_txt(txt_path, images_list[i],bbox,"bad")
        else:
            bbox = bbox_transfer(bbox,height,width)#(269,47,62,80)
            # bbox = (269,47,62,80)
            tracker.init(frame, bbox)

            while i<len(images_list):
                frame = cv2.imread(os.path.join(img_path, images_list[i]))

                timer = cv2.getTickCount()

                # Update tracker
                ok, bbox = tracker.update(frame)
                bbox_w = bbox_int(bbox)
                write_txt(txt_path, images_list[i], bbox_w)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    i = i + 1
                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure, re-initialize ...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                                2)



                # Display tracker type on frame
                cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

                # Display result
                cv2.imshow("Tracking", frame)

                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27: break

            break





        # while True:
        # 	if score > 0.7:
        # 		break
        # 	ok, frame = video.read()
        # 	bbox, score = bbox_score(frame)
        # 	bbox = np.squeeze(bbox)[0]
        # 	score = np.squeeze(score)[0]

        # bbox = bbox_transfer(bbox, height, width)


        # Initialize tracker with first frame and bounding box
        # ok = tracker.init(frame, bbox)
        # # bbox = bbox_transfer(bbox)
        #
        # while True:
        # 	# Read a new frame
        # 	# ok, frame = video.read()  # I think this line should be put after tracker.update
        # 	# if not ok:
        # 	# 	break
        #
        # 	# Start timer
        # 	timer = cv2.getTickCount()
        #
        # 	# Update tracker
        # 	ok, bbox = tracker.update(frame)
        #
        # 	# Calculate Frames per second (FPS)
        # 	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #
        # 	# Draw bounding box
        # 	if ok:
        # 		# Tracking success
        # 		p1 = (int(bbox[0]), int(bbox[1]))
        # 		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        # 		cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        # 	else:
        # 		# Tracking failure
        # 		cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        #
        # 	# Display tracker type on frame
        # 	cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        #
        # 	# Display FPS on frame
        # 	cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        #
        # 	# Display result
        # 	cv2.imshow("Tracking", frame)
        #
        # 	# Exit if ESC pressed
        # 	k = cv2.waitKey(1) & 0xff
        # 	if k == 27: break


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
