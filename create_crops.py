import numpy as np
import cv2
import scipy.io
import argparse
from utils import *
from tqdm import tqdm


def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
												 "and creates database for training.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# parser.add_argument("--output", "-o", type=str,default='train_data.npz',
	# 					help="path to output database mat file")
	parser.add_argument("--img_pth", type=str, default="/home/wei/Documents/DATA/kinship/ksframes/",
						help="dataset from videos")
	parser.add_argument("--img_size", type=int, default=64,
						help="output image size")
	# parser.add_argument("--min_score", type=float, default=1.0,
	#                     help="minimum face_score")
	args = parser.parse_args()
	return args


def enlarge_bbox(bbox,imgw,imgh,scale = 1.2):
	w = bbox[2]
	h = bbox[3]
	xcen = bbox[0]+w/2
	ycen = bbox[1]+h/2
	if xcen- w*scale/2>0:
		x1new = xcen- w*scale/2
	else:
		x1new = 0
	if ycen - h*scale/2>0:
		y1new = ycen - h*scale/2
	else:
		y1new = 0
	if xcen+ w*scale/2 < imgw:
		x2new = xcen+w*scale/2
	else:
		x2new = imgw
	if ycen + h*scale/2<imgh:
		y2new = ycen +h*scale/2
	else:
		y2new = imgh

	return int(x1new),int(y1new),int(x2new),int(y2new)

def main():
	args = get_args()
	# output_path = args.output
	img_pth = args.img_pth
	img_size = args.img_size
	im_g_pths = [os.path.join(img_pth,ls) for ls in sorted(os.listdir(img_pth))]


	out_ages = []
	out_imgs = []

	for kins in tqdm(im_g_pths):
		for kins_g in sorted(os.listdir(kins)):
			kg_path = os.path.join(kins,kins_g)
			cp_path = kg_path.replace('ksframes','croped')
			if not os.path.isdir(cp_path):
				os.makedirs(cp_path)
			img_pths = [os.path.join(kg_path,ls) for ls in sorted(os.listdir(kg_path))]

			for i, img_path in enumerate(img_pths):
				img = cv2.imread(img_path)
				bbox = read_bbox(img_path.replace('ksframes','bbox')[:-3]+'txt')
				px1,py1,px2,py2 = enlarge_bbox(bbox,img.shape[1],img.shape[0],1.35)
				# crop_img = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
				crop_img = img[py1:py2, px1:px2]
				## for debug
				# cv2.imshow("crop face", crop_img)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# out_imgs.append(cv2.resize(crop_img, (img_size, img_size)))
				temp = cv2.resize(crop_img,(img_size,img_size))
				outim_pth = img_path.replace('ksframes','croped')
				cv2.imwrite(outim_pth,temp)
				# out_ages.append(int(ages[i]))


	# np.savez(output_path,image=np.array(out_imgs), age=np.array(out_ages), img_size=img_size)

if __name__ == '__main__':
	main()

