import os
def bbox_transfer(bbox, height, width):
	b0 = int(bbox[1] * width)
	b1 = int(bbox[0] * height)
	b2 = int((bbox[3] - bbox[1]) * width)
	b3 = int((bbox[2] - bbox[0]) * height)
	# bbox = (bbox[1] * width, bbox[0] * height, bbox[2] - bbox[0], bbox[3] - bbox[1])
	return (b0, b1, b2, b3)

def bbox_int(bbox):

	b0 = int(bbox[0] )
	b1 = int(bbox[1] )
	b2 = int(bbox[2] )
	b3 = int(bbox[3] )
	return [b0,b1,b2,b3]

def write_txt(path,name,bbox,flag = "good"):

	txt_name = name.split(".")[0]+".txt"
	if flag == "bad":
		txt_name = name.split(".")[0] + "_bad.txt"
	path_name = os.path.join(path,txt_name)
	with open(path_name,"w") as ff:
		ff.write(''.join(str(b)+" " for b in bbox))

def read_txt(path,name):

	path_name = os.path.join(path,name)
	with open(path_name,"r") as ff:
		bbox = ff.readline().replace("\n","")
	return bbox


###############

def sort_img(iml):
	# for im_n in iml:
	# 	im_new = im_n.split(".")[0].replace("frame","")
	iml_new = [int(im_n.split(".")[0].replace("frame","")) for im_n in iml]
	#indx = np.argsort(iml_new)
	new_sort = sorted(list(zip(iml_new,iml)))
	iml_sort = list(list(zip(*new_sort))[1])
	return iml_sort

def check_file(filenames):

	f_type = ['.jpg','.png']
	for item in filenames:
		for types in f_type:
			if types in item:
				return True

	return False


def write_imglist(path_name,img_list):

	with open(path_name,"w") as ff:
		for imgs in img_list:
			ff.write(imgs[0]+'\t'+imgs[1]+'\n' )



def read_imglist(path_name):

	img_path = []
	ages = []
	# out_tuple = []
	# img_list = []
	with open(path_name,"r") as ff:
		img_list = ff.readlines()
	for imgl in img_list:
		img_path.append(imgl.split('\t')[0])
		age = imgl.split('\t')[-1].replace('\n','')
		ages.append(age)
		# out_tuple.append((img_path,age))

	return img_path, ages


def read_bbox(path_name):

	with open(path_name,'r') as ff:
		bbox = [int(b) for b in ff.readline().split(' ')[:-1]]
	return bbox


def path_classify(total_path):

	img_list = []

	for dirname, dirnames, filenames in os.walk(total_path):

		if not filenames:
			continue

		if not check_file(filenames):
			continue

		age = dirname.split('/')[-2]
		# age =[]
		# for items in dirnames:
		# 	if "__" in items:
		# 		age = dirname.split('/')[-1]
		filenames = sort_img(filenames)
		for imgs in filenames:

			img_path = os.path.join(dirname,imgs)
			img_list.append((img_path,age))

	return img_list



if __name__ == "__main__":
	## test 1

	img_list = path_classify('./data_try/')
	write_imglist('t1.lst',img_list)
	img_paths, ages = read_imglist('t1.lst')

	## test 2
	bbox = read_bbox('./txt/data_try/917/61/0__0/frame6.txt')
	print(bbox)

