import setting as st
import setting_2 as fst
import nibabel as nib
import numpy as np
import utils

""" load segment tempalte """
template_dir = st.template_dir
template = nib.load(template_dir).get_data().squeeze()
# np.unique(template) # 0, (10 : csf) , (150 : GM), (250 : WM)
list_minmax = [[np.inf, -1] for _ in range(3)]
# for i in range(256):
#     for j in range(256):
#         print(list_minmax)
#         for k in range(256):
#             if template[i, j, k] != 0:
#                 """ max """
#                 if list_minmax[0][1] < i :
#                     list_minmax[0][1] = i
#                 elif list_minmax[1][1] < j :
#                     list_minmax[1][1] = j
#                 elif list_minmax[2][1] < k :
#                     list_minmax[2][1] = k
#
#                 """ min """
#                 if list_minmax[0][0] > i :
#                     list_minmax[0][0] = i
#                 elif list_minmax[1][0] > j :
#                     list_minmax[1][0] = j
#                 elif list_minmax[2][0] > k :
#                     list_minmax[2][0] = k


# utils.save_featureMap_numpy(template[list_minmax[0][1]:list_minmax[0][0]+1, list_minmax[1][1]:list_minmax[1][0]+1, list_minmax[2][1]:list_minmax[2][0]+1], dirToSave='./test_11', name='tempalte')
# print(template[list_minmax[0][0]:list_minmax[0][1]+1, list_minmax[1][0]:list_minmax[1][1]+1, list_minmax[2][0]:list_minmax[2][1]+1].shape)


list_dir_all_modality = []
GM_sub_list = []
included_file_name_GM = ['gm']
for cnt, dir in enumerate(st.dir_list):
    GM_sub_list.append(
        utils.search_in_whole_subdir(file_dir=st.orig_data_dir, sub_dir=dir, n_file=included_file_name_GM,
                                     n_ext='.img'))
    print(len(GM_sub_list[cnt]))

"""
AD : 229
MCI : 403
AD : 198
sMCI : 214
pMCI : 160
"""
count = 0
list_MCI = []
for i in range(len(GM_sub_list[1])):
    cur_count = count
    for j in range(len(GM_sub_list[3])):
        if GM_sub_list[1][i][-25:-14] == GM_sub_list[3][j][-25:-14]:
            count += 1
            list_MCI.append(GM_sub_list[1][i])
    for k in range(len(GM_sub_list[4])):
        if GM_sub_list[1][i][-25:-14] == GM_sub_list[4][k][-25:-14]:
            count += 1
            list_MCI.append(GM_sub_list[1][i])
    if cur_count == count:
        print(GM_sub_list[1][i])

GM_sub_list.pop(1)
GM_sub_list.insert(1, list_MCI)

list_dir_all_modality.append(GM_sub_list)

""" allocate the memory  """
list_image_memalloc = []
list_age_memallow = []
list_MMSE_memallow = []

""" the # of the subject depending on the disease label """
n_NC_subjects = len(list_dir_all_modality[0][0])
n_MCI_subjects = len(list_dir_all_modality[0][1])
n_AD_subjects = len(list_dir_all_modality[0][2])
n_sMCI_subjects = len(list_dir_all_modality[0][3])
n_pMCI_subjects = len(list_dir_all_modality[0][4])

list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects, n_sMCI_subjects, n_pMCI_subjects]

""" save the data """
for i_modality in range(len(list_dir_all_modality)):
    for j_class in range(len(list_dir_all_modality[i_modality])):
        for k in range(len(list_dir_all_modality[i_modality][j_class])):
            tmp_dir_file = list_dir_all_modality[i_modality][j_class][k]
            print(tmp_dir_file[-24:-15])
            print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, k))
            tmp_image = np.squeeze(nib.load(tmp_dir_file).get_data())

            for i in range(256):
                for j in range(256):
                    print(list_minmax)
                    for k in range(256):
                        if tmp_image[i, j, k] != 0:
                            """ max """
                            if list_minmax[0][1] < i:
                                list_minmax[0][1] = i
                            elif list_minmax[1][1] < j:
                                list_minmax[1][1] = j
                            elif list_minmax[2][1] < k:
                                list_minmax[2][1] = k

                            """ min """
                            if list_minmax[0][0] > i:
                                list_minmax[0][0] = i
                            elif list_minmax[1][0] > j:
                                list_minmax[1][0] = j
                            elif list_minmax[2][0] > k:
                                list_minmax[2][0] = k


print(list_minmax)

print("finished!")
