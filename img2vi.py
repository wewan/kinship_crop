import cv2
import os
import numpy as np

img_folder = "data_try/0__0"
video_name = 'try.avi'

def sort_img(iml):
	# for im_n in iml:
	# 	im_new = im_n.split(".")[0].replace("frame","")
	iml_new = [int(im_n.split(".")[0].replace("frame","")) for im_n in iml]
	#indx = np.argsort(iml_new)
	new_sort = sorted(list(zip(iml_new,iml)))
	iml_sort = list(list(zip(*new_sort))[1])
	return iml_sort

def img2vi(img_folder,video_name):
	images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
	images = sort_img(images)
	frame = cv2.imread(os.path.join(img_folder, images[0]))

	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name,0,10,(width,height))

	for image in images:
		video.write(cv2.imread(os.path.join(img_folder,image)))

	cv2.destroyAllWindows()
	video.release()

if __name__ == '__main__':

	img2vi(img_folder,video_name)
