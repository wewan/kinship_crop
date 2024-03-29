import  os
from random import shuffle
import pickle


# a= sorted(os.listdir('/home/wei/Documents/DATA/kinship/ksframes/bro-bro'))
# print(a)

bro_bro = [('B-B_01_1', 'B-B_01_2'),
             ('B-B_02_1', 'B-B_02_2'),
             ('B-B_03_1', 'B-B_03_2'),
             ('B-B_04_1', 'B-B_04_2'),
             ('B-B_05_1', 'B-B_05_2'),
             ('B-B_05_3', 'B-B_05_2'),
             ('B-B_05_1', 'B-B_05_3'),
             ('B-B_06_1', 'B-B_06_2'),
             ('B-B_07_1', 'B-B_07_2'),
             ('B-B_08_1', 'B-B_08_2'),
             ('B-B_09_1', 'B-B_09_2'),
             ('B-B_10_1', 'B-B_10_2'),
             ('B-B_11_1', 'B-B_11_2'),
             ('B-B_12_1', 'B-B_12_2'),
             ('B-B_13_1', 'B-B_13_2')]

bro_sist =  [('B-S_01_1', 'B-S_01_2'),
             ('B-S_02_1', 'B-S_02_2'),
             ('B-S_03_1', 'B-S_03_2'),
             ('B-S_04_1', 'B-S_04_2'),
             ('B-S_05_1', 'B-S_05_2'),
             ('B-S_06_1', 'B-S_06_2'),
             ('B-S_07_1', 'B-S_07_2'),
             ('B-S_08_1', 'B-S_08_2'),
             ('B-S_09_1', 'B-S_09_2'),
             ('B-S_10_1', 'B-S_10_2'),
             ('B-S_11_1', 'B-S_11_2'),
             ('B-S_12_1', 'B-S_12_2'),
             ('B-S_12_3', 'B-S_12_2'),
             ('B-S_13_1', 'B-S_13_2'),
             ('B-S_14_1', 'B-S_14_2'),
             ('B-S_15_1', 'B-S_15_2'),
             ('B-S_16_1', 'B-S_16_2'),
             ('B-S_17_1', 'B-S_17_2'),
             ('B-S_18_1', 'B-S_18_2'),
             ('B-S_19_1', 'B-S_19_2'),
             ('B-S_20_1', 'B-S_20_2'),
             ('B-S_21_1', 'B-S_21_2'),
             ('B-S_22_1', 'B-S_22_2'),
             ('B-S_23_1', 'B-S_23_2'),
             ('B-S_24_1', 'B-S_24_2'),
             ('B-S_24_1', 'B-S_24_3'),
             ('B-S_25_1', 'B-S_25_2'),
             ('B-S_26_1', 'B-S_26_2'),
             ('B-S_27_1', 'B-S_27_2'),
             ('B-S_27_3', 'B-S_27_2')]


sist_sist =  [('S-S_01_1', 'S-S_01_2'),
              ('S-S_02_1', 'S-S_02_2'),
              ('S-S_03_1', 'S-S_03_2'),
              ('S-S_04_1', 'S-S_04_2'),
              ('S-S_05_1', 'S-S_05_2'),
              ('S-S_06_1', 'S-S_06_2'),
              ('S-S_07_1', 'S-S_07_2'),
              ('S-S_08_1', 'S-S_08_2'),
              ('S-S_09_1', 'S-S_09_2'),
              ('S-S_10_1', 'S-S_10_2'),
              ('S-S_11_1', 'S-S_11_2'),
              ('S-S_12_1', 'S-S_12_2'),
              ('S-S_13_1', 'S-S_13_2'),
              ('S-S_14_1', 'S-S_14_2'),
              ('S-S_15_1', 'S-S_15_2')]


father_son = [('F-S_01_1', 'F-S_01_2'),
              ('F-S_02_1', 'F-S_02_2'),
              ('F-S_03_1', 'F-S_03_2'),
              ('F-S_03_1', 'F-S_03_3'),
              ('F-S_04_1', 'F-S_04_2'),
              ('F-S_05_1', 'F-S_05_2'),
              ('F-S_06_1', 'F-S_06_2'),
              ('F-S_07_1', 'F-S_07_2'),
              ('F-S_08_1', 'F-S_08_2'),
              ('F-S_08_1', 'F-S_08_3'),
              ('F-S_09_1', 'F-S_09_2'),
              ('F-S_10_1', 'F-S_10_2'),
              ('F-S_11_1', 'F-S_11_2'),
              ('F-S_11_1', 'F-S_11_3'),
              ('F-S_12_1', 'F-S_12_2'),
              ('F-S_13_1', 'F-S_13_2'),
              ('F-S_13_1', 'F-S_13_3'),
              ('F-S_14_1', 'F-S_14_2'),
              ('F-S_15_1', 'F-S_15_2'),
              ('F-S_16_1', 'F-S_16_2'),
              ('F-S_17_1', 'F-S_17_2'),
              ('F-S_17_1', 'F-S_17_3'),
              ('F-S_18_1', 'F-S_18_2'),
              ('F-S_19_1', 'F-S_19_2'),
              ('F-S_20_1', 'F-S_20_2'),
              ('F-S_21_1', 'F-S_21_2'),
              ('F-S_21_1', 'F-S_21_3'),
              ('F-S_22_1', 'F-S_22_2'),
              ('F-S_23_1', 'F-S_23_2'),
              ('F-S_24_1', 'F-S_24_2'),
              ('F-S_25_1', 'F-S_25_2'),
              ('F-S_25_1', 'F-S_25_3'),
              ('F-S_26_1', 'F-S_26_2'),
              ('F-S_26_1', 'F-S_26_3')]

father_dau = [('F-D_01_1', 'F-D_01_2'),
              ('F-D_02_1', 'F-D_02_2'),
              ('F-D_03_1', 'F-D_03_2'),
              ('F-D_04_1', 'F-D_04_2'),
              ('F-D_05_1', 'F-D_05_2'),
              ('F-D_06_1', 'F-D_06_2'),
              ('F-D_07_1', 'F-D_07_2'),
              ('F-D_08_1', 'F-D_08_2'),
              ('F-D_08_1', 'F-D_08_3'),
              ('F-D_09_1', 'F-D_09_2'),
              ('F-D_10_1', 'F-D_10_2'),
              ('F-D_11_1', 'F-D_11_2'),
              ('F-D_12_1', 'F-D_12_2'),
              ('F-D_13_1', 'F-D_13_2'),
              ('F-D_14_1', 'F-D_14_2'),
              ('F-D_14_1', 'F-D_14_3'),
              ('F-D_15_1', 'F-D_15_2'),
              ('F-D_16_1', 'F-D_16_2'),
              ('F-D_17_1', 'F-D_17_2'),
              ('F-D_18_1', 'F-D_18_2'),
              ('F-D_19_1', 'F-D_19_2'),
              ('F-D_20_1', 'F-D_20_2'),
              ('F-D_21_1', 'F-D_21_2'),
              ('F-D_22_1', 'F-D_22_2'),
              ('F-D_22_1', 'F-D_22_3'),
              ('F-D_23_1', 'F-D_23_2'),
              ('F-D_23_1', 'F-D_23_3'),
              ('F-D_24_1', 'F-D_24_2'),
              ('F-D_25_1', 'F-D_25_2'),
              ('F-D_25_1', 'F-D_25_3')]

mother_dau = [('M-D_01_1', 'M-D_01_2'),
              ('M-D_02_1', 'M-D_02_2'),
              ('M-D_03_1', 'M-D_03_2'),
              ('M-D_04_1', 'M-D_04_2'),
              ('M-D_05_1', 'M-D_05_2'),
              ('M-D_05_1', 'M-D_05_3'),
              ('M-D_06_1', 'M-D_06_2'),
              ('M-D_07_1', 'M-D_07_2'),
              ('M-D_08_1', 'M-D_08_2'),
              ('M-D_09_1', 'M-D_09_2'),
              ('M-D_10_1', 'M-D_10_2'),
              ('M-D_11_1', 'M-D_11_2'),
              ('M-D_12_1', 'M-D_12_2'),
              ('M-D_13_1', 'M-D_13_2'),
              ('M-D_14_1', 'M-D_14_2'),
              ('M-D_15_1', 'M-D_15_2'),
              ('M-D_16_1', 'M-D_16_2'),
              ('M-D_17_1', 'M-D_17_2'),
              ('M-D_18_1', 'M-D_18_2'),
              ('M-D_19_1', 'M-D_19_2'),
              ('M-D_20_1', 'M-D_20_2'),
              ('M-D_21_1', 'M-D_21_2'),
              ('M-D_22_1', 'M-D_22_2'),
              ('M-D_22_1', 'M-D_22_3'),
              ('M-D_23_1', 'M-D_23_2'),
              ('M-D_24_1', 'M-D_24_2'),
              ('M-D_24_1', 'M-D_24_3'),
              ('M-D_25_1', 'M-D_25_2'),
              ('M-D_26_1', 'M-D_26_2'),
              ('M-D_26_1', 'M-D_26_3'),
              ('M-D_27_1', 'M-D_27_2'),
              ('M-D_28_1', 'M-D_28_2'),
              ('M-D_29_1', 'M-D_29_2'),
              ('M-D_30_1', 'M-D_30_2'),
              ('M-D_31_1', 'M-D_31_2'),
              ('M-D_31_3', 'M-D_31_1'),
              ('M-D_32_1', 'M-D_32_2'),
              ('M-D_33_1', 'M-D_33_2'),
              ('M-D_34_1', 'M-D_34_2'),
              ('M-D_35_1', 'M-D_35_2'),
              ('M-D_35_1', 'M-D_35_3'),
              ('M-D_36_1', 'M-D_36_2'),
              ('M-D_36_1', 'M-D_36_3'),
              ('M-D_37_1', 'M-D_37_2'),
              ('M-D_37_1', 'M-D_37_3'),
              ('M-D_38_1', 'M-D_38_2')]

mother_son = [('M-S_01_1', 'M-S_01_2'),
              ('M-S_02_1', 'M-S_02_2'),
              ('M-S_03_1', 'M-S_03_2'),
              ('M-S_04_1', 'M-S_04_2'),
              ('M-S_05_1', 'M-S_05_2'),
              ('M-S_06_1', 'M-S_06_2'),
              ('M-S_07_1', 'M-S_07_2'),
              ('M-S_08_1', 'M-S_08_2'),
              ('M-S_08_1', 'M-S_08_3'),
              ('M-S_09_1', 'M-S_09_2'),
              ('M-S_10_1', 'M-S_10_2'),
              ('M-S_11_1', 'M-S_11_2'),
              ('M-S_12_1', 'M-S_12_2'),
              ('M-S_12_1', 'M-S_12_3'),
              ('M-S_13_1', 'M-S_13_2'),
              ('M-S_14_1', 'M-S_14_2'),
              ('M-S_14_1', 'M-S_14_3'),
              ('M-S_15_1', 'M-S_15_2'),
              ('M-S_16_1', 'M-S_16_2'),
              ('M-S_16_1', 'M-S_16_3'),
              ('M-S_17_1', 'M-S_17_2'),
              ('M-S_18_1', 'M-S_18_2'),
              ('M-S_19_1', 'M-S_19_2'),
              ('M-S_20_1', 'M-S_20_2'),
              ('M-S_21_1', 'M-S_21_2'),
              ('M-S_22_1', 'M-S_22_2'),
              ('M-S_23_1', 'M-S_23_2'),
              ('M-S_23_1', 'M-S_23_3'),
              ('M-S_24_1', 'M-S_24_2'),
              ('M-S_25_1', 'M-S_25_2'),
              ('M-S_26_1', 'M-S_26_2'),
              ('M-S_27_1', 'M-S_27_2'),
              ('M-S_27_1', 'M-S_27_3'),
              ('M-S_28_1', 'M-S_28_2'),
              ('M-S_29_1', 'M-S_29_2'),
              ('M-S_30_1', 'M-S_30_2'),
              ('M-S_31_1', 'M-S_31_2'),
              ('M-S_32_1', 'M-S_32_2'),
              ('M-S_33_1', 'M-S_33_2'),
              ('M-S_33_1', 'M-S_33_3'),
              ('M-S_34_1', 'M-S_34_2')]


kin_dict = {'bro_bro':bro_bro,'bro_sist':bro_sist,
            'sist_sist':sist_sist,'father_son':father_son,
            'mother_dau':mother_dau,'mother_son':mother_son,
            'father_dau':father_dau}


def generate_ls(ls,split_n):
    """
    :param ls:
    :param split_n: split into n folds
    :return:
    """
    sn = split_n
    lth_f = int(len(ls)/sn)
    train_ls = []
    shuffle(ls)
    for i in range(sn):
        if i != sn-1:
            a = ls[i*lth_f:i*lth_f+lth_f]
            # print(a)
        else:
            a = ls[i*lth_f:]
            # print(a)
        a_neg = _generate_neg(a)

        for im1,im2 in a:
            train_ls.append([i+1,1,im1,im2])

        for im1,im2 in a_neg:
            train_ls.append([i+1,0,im1,im2])

    # print(train_ls)
    return train_ls




def _generate_neg(ls):
    """
    generate neg pairs
    :param ls:
    :return:
    """
    it_1 = []
    it_2 = []
    for i in ls:
        it_1.append(i[0])
        it_2.append(i[1])

    while _check(it_1,it_2):
        shuffle(it_2)

    neg_ls = list(zip(it_1,it_2))
    # print(neg_ls)
    return neg_ls


def _check(a,b):
    """
    if there is more than one pair matched, return True
    :param a:
    :param b:
    :return:
    """
    for i,j in zip(a,b):
        if i.split('_')[1]==j.split('_')[1]:
            return True
    return False



if __name__=='__main__':

    for kin_g in kin_dict:

        train_ls = generate_ls(kin_dict[kin_g], 5)
        with open('kin_ls/{}.pkl'.format(kin_g), 'wb') as fp:
            pickle.dump(train_ls, fp)

    # with open ('father_dau.pkl', 'rb') as fp:
    #     itemlist = pickle.load(fp)
    # print(8)

