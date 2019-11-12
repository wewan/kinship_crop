"""
extract frames from original segmented videos
"""

import cv2
import sys
import os
import pandas as pd
from tqdm import tqdm

kin_dict = {'B-B':'bro-bro','B-S':'bro-sist','S-S':'sist-sist','F-D':'father-dau',
            'F-S':'father-son','M-D':'mother-dau','M-S':'mother-son'}

def _extract_frm(ori_pth, dir_pth):
    vidcap = cv2.VideoCapture(ori_pth)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("{}/frame{:03d}.jpg".format(dir_pth, count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    # print("finish one")

def extract_frm(ipth,opth):
    """
    :param ipth: ori videos list for extracting
    :param opth: out put path
    :return: saving the extracted frames in opth
    """
    for vis in sorted(os.listdir(ipth)):
        if vis.startswith('4'):
            ovpth = os.path.join(ipth, vis)
            # dpth = os.path.join(opth, folds)
            if not os.path.isdir(opth):
                os.makedirs(opth)
            _extract_frm(ovpth, opth)



def extract_frm_allyes(ipth,opth):
    """
    extrat all yes from 0/2/4/6.mp4
    :param ipth: ori videos list for extracting
    :param opth: out put path
    :return: saving the extracted frames in opth
    """
    for vis in sorted(os.listdir(ipth)):
        for i in ['0','2','4','6']:
            if vis.startswith(i):
                ovpth = os.path.join(ipth, vis)
                dpth = os.path.join(opth, i)
                if not os.path.isdir(dpth):
                    os.makedirs(dpth)
                _extract_frm(ovpth, dpth)



def read_xls(xlsx):
    a = pd.read_excel(xlsx)


# todo: there is a bug in it e.g. 42.1;42.2 42.1;42.4 42.2;42.4
if __name__ == '__main__':
    # ori_path = '/home/wei/Documents/DATA/kinship/sp_video/vi-4-21/'
    # dir_path = '/home/wei/Documents/DATA/kinship/ksframes/vi-4-21/'
    # for i, folds in enumerate(sorted(os.listdir(ori_path))):
    #     if i ==9:
    #         opth = os.path.join(ori_path, folds)
    #         for vis in sorted(os.listdir(opth)):
    #             if vis.startswith('4'):
    #                 ovpth = os.path.join(opth,vis)
    #                 dpth = os.path.join(dir_path, folds)
    #                 if not os.path.isdir(dpth):
    #                   os.makedirs(dpth)
    #                 _extract_frm(ovpth, dpth)
    print(6)

    # load path

    ori_path = '/home/wei/Documents/Data/kinship/crops_ori/'
    dir_path = '/home/wei/Documents/Data/kinship/ksframes/'

    give_kin = "F-D"
    ori_ud = os.path.join(ori_path,kin_dict[give_kin])
    dir_ud = os.path.join(dir_path,kin_dict[give_kin])
    a = pd.read_excel('kinship.xlsx')

    def get_len(b):
        lth = len(b)
        for i in range(lth):
            if str(b[i])=='nan':
                return i
        return lth

    # not_in = lambda a,dic: dic[[item for item in dic.keys() if item != a][0]]
    # add_one = lambda str_b: str_b[:-1]+ str(int(str_b[-1])+1)
    def get_bfr(dic):
        base = list(dic.values())[0][:-1]
        num1 = int(list(dic.values())[0][-1])
        num2 = int(list(dic.values())[1][-1])
        num = max([num1,num2])
        return base+str(num+1)

    b = a[give_kin]
    lth = get_len(b)
    temp = {}
    num_kin = 0
    train_ls = 0
    kinship_train_list = []
    # loop all kinship pair in b
    for id in tqdm(range(lth)):

        # load pair a,b
        kin_a = b[id].split(';')[0].replace('.','-')
        kin_b = b[id].split(';')[1].replace('.','-')
        kin_a_pth = os.path.join(ori_ud,kin_a)
        kin_b_pth = os.path.join(ori_ud,kin_b)

        # generate a_fr,b_fr name for saving
        # if a is the same with temp, a_fr keep the same b_fr+1

        if kin_a in temp:
            kin_a_fr = temp[kin_a]
            kin_b_fr = get_bfr(temp)
            temp = {kin_a:kin_a_fr,kin_b:kin_b_fr}
        elif kin_b in temp:
            kin_a_fr = get_bfr(temp)
            kin_b_fr = temp[kin_b]
            temp = {kin_a: kin_a_fr, kin_b: kin_b_fr}
        else:
            num_kin += 1
            kin_a_fr = '{}_{:02d}_{}'.format(give_kin,num_kin,1)
            kin_b_fr = '{}_{:02d}_{}'.format(give_kin,num_kin,2)
            temp = {kin_a: kin_a_fr, kin_b: kin_b_fr}
        # extra_frm function e_f(a,a_fr) e_f(b,b_fr)
        kinship_train_list.append((kin_a_fr,kin_b_fr))
        afr_pth = os.path.join(dir_ud, kin_a_fr)
        bfr_pth = os.path.join(dir_ud, kin_b_fr)
        extract_frm_allyes(kin_a_pth,afr_pth)
        extract_frm_allyes(kin_b_pth,bfr_pth)

    print(kinship_train_list)









#