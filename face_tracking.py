from ft_img import *
from face_detect import *
from tqdm import tqdm
def check_file(filenames):

	f_type = ['.jpg','.png']
	for item in filenames:
		for types in f_type:
			if types in item:
				return True

	return False

def face_tracking(total_path,mode = 1):
	for dirname, dirnames, filenames in os.walk(total_path):
		if not filenames:
			continue

		if not check_file(filenames):
			continue

		if mode == 1:
			tracking_face(dirname)
		elif mode == 2:
			track_ssd(dirname)


	# path_lists = os.listdir(total_path)
	# print(name for name in os.listdir(total_path) if os.path.isdir(name))

if __name__ == "__main__":
	"""
	mode 1: ssd detection + opencv
	mode 2: pure ssd detection
	"""
	path = "/home/wei/Documents/DATA/kinship/ksframes/"
	ls = sorted(os.listdir(path))
	for sl in tqdm(ls):
		sub_path = os.path.join(path,sl)
		for su in tqdm(sorted(os.listdir(sub_path))):
			face_tracking(path,mode = 2)
