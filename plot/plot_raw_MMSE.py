import numpy as np
import setting as st
from data_load import data_load as DL
import torch
import torch.nn as nn
import utils as ut
import matplotlib.pyplot as plt
from matplotlib import gridspec

def test(config, fold, dir_MMSE_dist, flag_estimation = True):
    """ free all GPU memory """
    torch.cuda.empty_cache()

    """ load data """ # image, lbl, age, MMSE
    list_train_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_total, flag_tr_val_te='train')
    list_val_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_total, flag_tr_val_te='val')
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_total, flag_tr_val_te='test')

    if flag_estimation == True:
        for i in range(len(st.list_selected_for_total)):
            lambda_disease_factor = st.list_selected_lambdas_at_age[i]
            list_train_data[2][(list_train_data[1] == i)] = ut.estimate_biological_age(
                age=list_train_data[2][(list_train_data[1] == i)],
                MMSE=list_train_data[3][(list_train_data[1] == i)],
                lambda_disease_factor=lambda_disease_factor)

            list_val_data[2][(list_val_data[1] == i)] = ut.estimate_biological_age(
                age=list_val_data[2][(list_val_data[1] == i)],
                MMSE=list_val_data[3][(list_val_data[1] == i)],
                lambda_disease_factor=lambda_disease_factor)

            list_test_data[2][(list_test_data[1] == i)] = ut.estimate_biological_age(
                age=list_test_data[2][(list_test_data[1] == i)],
                MMSE=list_test_data[3][(list_test_data[1] == i)],
                lambda_disease_factor=lambda_disease_factor)

    fig = plt.figure(figsize=(len(st.list_class_for_total) * 12, 25))
    plt.rcParams.update({'font.size': 22})
    if flag_estimation == True :
        fig.suptitle('Estimated Age and MMSE distribution of fold{0} ({1})'.format(fold, st.list_age_estimating_function[st.selected_function]), fontsize=50)
    else:
        fig.suptitle('Labeled Age and MMSE distribution of fold{0}'.format(fold), fontsize=50)

    # plt.xticks([])

    heights = [10, 2, 10, 2, 10, 2]
    widths = []
    for i_tmp in range(len(st.list_class_for_total)):
        widths.append(10)
        widths.append(3)

    gs = gridspec.GridSpec(nrows=6,  # row
                           ncols=len(st.list_class_for_total) * 2,  # col
                           height_ratios=heights,
                           width_ratios=widths
                           )

    # max_age = max(np.hstack([list_train_data[2], list_val_data[2], list_test_data[2]]))
    # min_age = min(np.hstack([list_train_data[2], list_val_data[2], list_test_data[2]]))
    # max_MMSE = max(np.hstack([list_train_data[3], list_val_data[3], list_test_data[3]]))
    # min_MMSE = min(np.hstack([list_train_data[3], list_val_data[3], list_test_data[3]]))

    # age_left = min_age - 10
    # age_right = max_age + 10
    # MMSE_left = min_MMSE - 10
    # MMSE_right = max_MMSE + 10

    age_left = 50
    age_right = 110
    MMSE_left = 10
    MMSE_right = 40

    gap_1 = 2
    gap_2 = 5
    text_fontsize = 15

    """ loop for test, val, train in this order """
    list_loader = ['train', 'val', 'test']
    for i_loader, dataset in enumerate(list_loader):
        """ param for accuracy """
        if dataset == 'test':
            np_lbl = list_test_data[1]
            np_age = list_test_data[2]
            np_MMSE = list_test_data[3]
        elif dataset == 'val':
            np_lbl = list_val_data[1]
            np_age = list_val_data[2]
            np_MMSE = list_val_data[3]
        elif dataset == 'train':
            np_lbl = list_train_data[1]
            np_age = list_train_data[2]
            np_MMSE = list_train_data[3]

        np_age = np_age.squeeze()
        np_lbl = np_lbl.squeeze()
        np_MMSE = np_MMSE.squeeze()

        list_age_info = [] # classes (3,)
        list_MMSE_info = [] # classes (3,)
        for i in range(len(st.list_selected_for_total)):
            list_tmp_age = {}
            age_class_i = np_age[(np_lbl == i)]
            list_tmp_age['min'] = age_class_i.min()
            list_tmp_age['max'] = age_class_i.max()
            list_tmp_age['mean'] = age_class_i.mean()
            list_tmp_age['std'] = age_class_i.std()
            list_age_info.append(list_tmp_age)

            list_tmp_MMSE = {}
            MMSE_class_i = np_MMSE[(np_lbl == i)]
            list_tmp_MMSE['min'] = MMSE_class_i.min()
            list_tmp_MMSE['max'] = MMSE_class_i.max()
            list_tmp_MMSE['mean'] = MMSE_class_i.mean()
            list_tmp_MMSE['std'] = MMSE_class_i.std()
            list_MMSE_info.append(list_tmp_MMSE)



        for j_disease in range(len(st.list_selected_for_total)):
            # ax1 = plt.subplot(gs[0])
            row = i_loader * 2
            col = j_disease * 2
            ax1 = fig.add_subplot(gs[row, col])
            ax1.scatter(np_age[(np_lbl == j_disease)], np_MMSE[(np_lbl == j_disease)])

            ax1.set_title('{}  {}'.format(dataset, st.list_selected_for_total[j_disease]), fontsize=25)  # title of plot

            ax1.set_xlim([age_left, age_right])
            ax1.set_ylim([MMSE_left, MMSE_right])
            ax1.grid(True)

            ax1.set_ylabel('MMSE')
            ax1.set_xlabel('Age')

            # if col == 0:
            #     ax1.set_ylabel('Labeled MMSE')
            # else:
            #     ax1.set_yticks([])
            #
            # if row == 2:
            #     ax1.set_xlabel('Labeled age')
            # else:
            #     ax1.set_xticks([])


            ax1.text(age_right + 1, MMSE_right, 'age', fontsize=text_fontsize + 5)
            ax1.text(age_right + 1, MMSE_right - (1 * gap_1), 'min: {:.2f}'.format(list_age_info[j_disease]['min']), fontsize=text_fontsize)
            ax1.text(age_right + 1, MMSE_right - (2 * gap_1), 'max: {:.2f}'.format(list_age_info[j_disease]['max']), fontsize=text_fontsize)
            ax1.text(age_right + 1, MMSE_right - (3 * gap_1), 'mean: {:.2f}'.format(list_age_info[j_disease]['mean']), fontsize=text_fontsize)
            ax1.text(age_right + 1, MMSE_right - (4 * gap_1), 'std: {:.2f}'.format(list_age_info[j_disease]['std']), fontsize=text_fontsize)

            ax1.text(age_right + 1, MMSE_right - (4 * gap_1) - (1 * gap_1) - gap_2, 'MMSE', fontsize=text_fontsize + 5)
            ax1.text(age_right + 1, MMSE_right - (4 * gap_1) - (2 * gap_1) - gap_2, 'min: {:.2f}'.format(list_MMSE_info[j_disease]['min']), fontsize=text_fontsize)
            ax1.text(age_right + 1, MMSE_right - (4 * gap_1) - (3 * gap_1) - gap_2, 'max: {:.2f}'.format(list_MMSE_info[j_disease]['max']), fontsize=text_fontsize)
            ax1.text(age_right + 1, MMSE_right - (4 * gap_1) - (4 * gap_1) - gap_2, 'mean: {:.2f}'.format(list_MMSE_info[j_disease]['mean']), fontsize=text_fontsize)
            ax1.text(age_right + 1, MMSE_right - (4 * gap_1) - (5 * gap_1) - gap_2, 'std: {:.2f}'.format(list_MMSE_info[j_disease]['std']), fontsize=text_fontsize)

            """ save the figure """
            if flag_estimation == True:
                plt.savefig(dir_MMSE_dist + '/fold{}_estimated.png'.format(fold))
            else:
                plt.savefig(dir_MMSE_dist + '/fold{}_labeled.png'.format(fold))

    """ close all plot """
    plt.close('all')
