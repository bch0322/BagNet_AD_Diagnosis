import nibabel as nib
from matplotlib import gridspec
import numpy as np
import setting as st
from data_load import data_load as DL
import torch
from torch.autograd import Variable
import utils as ut
import matplotlib.pyplot as plt
import setting_2 as fst


def test(config, fold, model, dir_to_load, dir_age_pred):
    """ free all GPU memory """
    torch.cuda.empty_cache()

    """ load the fold list for test """
    list_train_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_total, flag_tr_val_te='train')
    list_val_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_total, flag_tr_val_te='val')
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_total, flag_tr_val_te='test')


    train_loader = DL.convert_Dloader_3(config.v_batch_size, list_train_data[0], list_train_data[1], list_train_data[2],
                                        list_train_data[3], is_training=False, num_workers=1, shuffle=False)
    val_loader = DL.convert_Dloader_3(config.v_batch_size, list_val_data[0], list_val_data[1], list_val_data[2],
                                      list_val_data[3], is_training=False, num_workers=1, shuffle=False)
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2],
                                       list_test_data[3], is_training=False, num_workers=1, shuffle=False)

    del list_train_data, list_val_data, list_test_data

    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    fig = plt.figure(figsize=(len(st.list_selected_for_total) * 12, 25))
    plt.rcParams.update({'font.size': 22})

    if fst.flag_estimate_age == True :
        fig.suptitle('Comparing between labeled and predicted ages in fold{0} ({1})'.format(fold, st.list_age_estimating_function[st.selected_function]), fontsize=50)
    else:
        fig.suptitle('Comparing between labeled and predicted ages in fold{0}'.format(fold), fontsize=50)

    # plt.xticks([])

    heights = [10, 2, 10, 2, 10, 2]
    widths = []
    for i_tmp in range(len(st.list_selected_for_total)):
        widths.append(10)
        widths.append(3)
    gs = gridspec.GridSpec(nrows=6,  # row
                           ncols=len(st.list_selected_for_total) * 2,  # col
                           height_ratios=heights,
                           width_ratios=widths
                           )
    age_left = 50
    age_right = 110
    pred_left = 50
    pred_right = 110
    gap_1 = 4
    gap_2 = 10
    text_fontsize = 15

    """ eval """
    list_loader = ['train', 'val', 'test']
    for i_loader, dataset in enumerate(list_loader):
        """ param for accuracy """
        if dataset == 'train':
            loader = train_loader
        elif dataset == 'val':
            loader = val_loader
        elif dataset == 'test':
            loader = test_loader

        """ param for accuracy """
        list_age = []
        list_lbl = []
        list_pred = []
        with torch.no_grad():
            for datas, labels, alabel, mlabel in loader:

                """ input"""
                datas = Variable(datas).cuda()
                labels = Variable(labels.long()).cuda()
                alabel = Variable(alabel.float()).cuda()

                """ forward propagation """
                dict_result = model(datas)
                pred_age = dict_result['preds']

                """ appending to the list """
                list_pred.append(pred_age.data.cpu().numpy().squeeze())
                list_lbl.append(labels.data.cpu().numpy().squeeze())
                list_age.append(alabel.data.cpu().numpy().squeeze())

        np_age = np.vstack(list_age).squeeze()
        np_lbl = np.vstack(list_lbl).squeeze()
        np_pred = np.vstack(list_pred).squeeze()

        for j_disease in range(len(st.list_selected_for_total)):
            row = i_loader * 2
            col = j_disease * 2
            ax1 = fig.add_subplot(gs[row, col])
            ax1.scatter(np_age[(np_lbl == j_disease)], np_pred[(np_lbl == j_disease)])
            ax1.plot(range(age_left, age_right), range(age_left, age_right))
            ax1.set_title('{}  {}'.format(dataset, st.list_selected_for_total[j_disease]), fontsize=25)  # title of plot

            ax1.set_xlim([age_left, age_right])
            ax1.set_ylim([pred_left, pred_right])
            ax1.grid(True)

            ax1.set_ylabel('predicted age')
            ax1.set_xlabel('labeled age')

            # if col == 0:
            #     ax1.set_ylabel('Labeled MMSE')
            # else:
            #     ax1.set_yticks([])
            #
            # if row == 2:
            #     ax1.set_xlabel('Labeled age')
            # else:
            #     ax1.set_xticks([])

            ax1.text(age_right + 1, pred_right, 'labeled age', fontsize=text_fontsize + 5)
            ax1.text(age_right + 1, pred_right - (1 * gap_1), 'min: {:.2f}'.format(np_age[(np_lbl == j_disease)].min()),
                     fontsize=text_fontsize)
            ax1.text(age_right + 1, pred_right - (2 * gap_1), 'max: {:.2f}'.format(np_age[(np_lbl == j_disease)].max()),
                     fontsize=text_fontsize)
            ax1.text(age_right + 1, pred_right - (3 * gap_1), 'mean: {:.2f}'.format(np_age[(np_lbl == j_disease)].mean()),
                     fontsize=text_fontsize)
            ax1.text(age_right + 1, pred_right - (4 * gap_1), 'std: {:.2f}'.format(np_age[(np_lbl == j_disease)].std()),
                     fontsize=text_fontsize)

            ax1.text(age_right + 1, pred_right - (4 * gap_1) - (1 * gap_1) - gap_2, 'pred age', fontsize=text_fontsize + 5)
            ax1.text(age_right + 1, pred_right - (4 * gap_1) - (2 * gap_1) - gap_2, 'min: {:.2f}'.format(np_pred[(np_lbl == j_disease)].min()),
                     fontsize=text_fontsize)
            ax1.text(age_right + 1, pred_right - (4 * gap_1) - (3 * gap_1) - gap_2, 'max: {:.2f}'.format(np_pred[(np_lbl == j_disease)].max()),
                     fontsize=text_fontsize)
            ax1.text(age_right + 1, pred_right - (4 * gap_1) - (4 * gap_1) - gap_2, 'mean: {:.2f}'.format(np_pred[(np_lbl == j_disease)].mean()),
                     fontsize=text_fontsize)
            ax1.text(age_right + 1, pred_right - (4 * gap_1) - (5 * gap_1) - gap_2, 'std: {:.2f}'.format(np_pred[(np_lbl == j_disease)].std()),
                     fontsize=text_fontsize)

            """ save the figure """
            plt.savefig(dir_age_pred + '/fold{}_age_prediction.png'.format(fold))

    """ close all plot """
    plt.close('all')