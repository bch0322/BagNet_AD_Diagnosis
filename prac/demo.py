import pandas as pd
import setting as st
import utils

""" tadpole dataset """
demo_dir = '../raven_label.csv'
data = pd.read_csv(demo_dir)


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

print("finish")

count = 0
for i in range(len(GM_sub_list[3])):
    for j in range(len(data['PATID'])):
        if GM_sub_list[3][i][-24:-14] == data['PATID'][j]:
            data['Group'][j] = 'sMCI'
            count += 1
print(count)

count = 0
for i in range(len(GM_sub_list[4])):
    for j in range(len(data['PATID'])):
        if GM_sub_list[4][i][-24:-14] == data['PATID'][j]:
            data['Group'][j] = 'pMCI'
            count += 1
print(count)
data.to_csv('./test_1.csv', header=True, index=False)

data['Group']