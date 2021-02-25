import nibabel as nib
import numpy as np
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import torch
from torch.autograd import Variable
import torch.nn as nn
import utils as ut
import os
from scipy import stats
from sklearn.metrics import confusion_matrix

def test(config, fold, model, dir_to_load,  dir_confusion):
    """ free all GPU memory """
    torch.cuda.empty_cache()
    criterion = nn.L1Loss()
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test,
                                                 flag_tr_val_te='test')
    """ load the fold list for test """
    if fst.flag_gaussian_norm == True:
        tmp_load_dir = st.gaussian_dir + '/fold_{}'.format(fold)
        npzfile = np.load(tmp_load_dir + '/mu_sigma.npz')
        mu = npzfile['mu']
        sigma = npzfile['sigma']
        list_test_data[0] = ut.Gauss_Norm_subjectWise(list_test_data[0], mu, sigma, train=False)

    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2],
                                       list_test_data[3], is_training=False, num_workers=1, shuffle=False)

    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    dict_result = ut.eval_regression_model(config, test_loader, model, criterion)


    return dict_result
