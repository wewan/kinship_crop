import cv2
import numpy as np
import math
from collections import defaultdict
from PIL import Image,ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from src import detect_faces
import torch


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))
def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')
        # draw.point((p[3],p[8]),fill= 'blue')
    return img_copy

def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    imshow(origin_img)


def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # image_array = np.array(image_array)
    image_array = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)

    # calculate the mean point of landmarks of left and right eye
    left_eye_center = (landmarks[0][0],landmarks[0][5])
    right_eye_center = (landmarks[0][1],landmarks[0][6])
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle




def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    landmarks_ls = list(np.zeros((1,10)))
    # rotated_landmarks = defaultdict(list)
    for i in range(5):
        rotated_landmark = rotate(origin=eye_center, point=(landmarks[0][i],landmarks[0][i+5]), angle=angle, row=row)
        landmarks_ls[0][i] = rotated_landmark[0]
        landmarks_ls[0][i+5] =  rotated_landmark[1]
    return landmarks_ls

def corp_face(image_array, size, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param size: single int value, size for w and h after crop
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    left, top: left and top coordinates of cropping

    take the eye's center to be the x center , nose's y to be the y center
    """
    x_center = (landmarks[0][0]+landmarks[0][1])/2
    # x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - size / 2, x_center + size / 2)

    y_center = landmarks[0][5]
    top, bottom = (y_center - size / 2, y_center + size /2)

    # pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = image_array.crop((left, top, right, bottom))
    # cropped_img = np.array(cropped_img)
    return cropped_img, left, top


def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    landmarks_ls = list(np.zeros((1, 10)))

    for i in range(5):
        landmarks_ls[0][i] = landmarks[0][i] - left
        landmarks_ls[0][i+5] = landmarks[0][i+5] -top

    return landmarks_ls

def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    imshow(origin_img)

if __name__=='__main__':

    # img_name = 'F-D_03_1/frame000.jpg'
    img_name = 'img/frame036.jpg'


    # PLimg = Image.open(img_name)
    image = cv2.imread(img_name)
    image  =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    PLimg = Image.fromarray(image)

    # Image.fromarray(image)
    # image.show()
    bounding_boxes, landmarks = detect_faces(PLimg)
    img_copy = show_bboxes(PLimg, bounding_boxes, landmarks)
    # img_copy.show()


    rotated_img,eye_center,angle= align_face(PLimg,landmarks)

    cv2.imwrite('rotated.jpg',rotated_img)
    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    rotated_PLimg = Image.fromarray(rotated_img)
    # rotated_PLimg.show()


    # # rotated_img.imshow()
    #
    #
    rotated_landmarks = rotate_landmarks(landmarks,eye_center,angle,rotated_PLimg.size[0])
    # print(landmarks)
    # print(rotated_landmarks)
    # rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    # rotated_img = Image.fromarray(rotated_img)
    img_copy = show_bboxes(rotated_PLimg, bounding_boxes, rotated_landmarks)
    # img_copy.show()

    cropped_img, left, top = corp_face(rotated_PLimg, 160, landmarks)
    cropped_img.show()


