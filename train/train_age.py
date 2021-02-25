import os
import sys
import utils as ut
from torch.autograd import Variable
import torch
import torch.nn as nn
import setting as st
import setting_2 as fst
from data_load import data_load as DL
from scipy import stats
import numpy as np
import nibabel as nib

def train(config, fold, model, optimizer, scheduler, list_dir_save_model, dir_pyplot, Validation=True, Test_flag = True):

    """ loss """
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # ES = ut.EarlyStopping(delta=0, patience=st.early_stopping_patience, verbose=True)
    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=st.early_stopping_patience, verbose=True))

    """ load data """
    list_train_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_train, flag_tr_val_te='train') # image, label, age, MMSE
    list_val_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test, flag_tr_val_te='val')
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test, flag_tr_val_te='test')

    if fst.flag_gaussian_norm == True:
        if fst.flag_gaussian_norm_init == True:
            """ save dir """
            tmp_save_dir = st.gaussian_dir + '/fold_{}'.format(fold)
            ut.make_dir(dir=tmp_save_dir, flag_rm=False)
            """ normalize """
            list_train_data[0], mu, sigma = ut.Gauss_Norm_subjectWise(list_train_data[0], train=True)
            np.savez(tmp_save_dir + '/mu_sigma', mu=mu, sigma=sigma)
            list_val_data[0] = ut.Gauss_Norm_subjectWise(list_val_data[0], mu, sigma, train=False)
            list_test_data[0] = ut.Gauss_Norm_subjectWise(list_test_data[0], mu, sigma, train=False)
        else:
            tmp_load_dir = st.gaussian_dir + '/fold_{}'.format(fold)
            npzfile = np.load(tmp_load_dir + '/mu_sigma.npz')
            mu = npzfile['mu']
            sigma = npzfile['sigma']
            list_train_data[0] = ut.Gauss_Norm_subjectWise(list_train_data[0], mu, sigma, train=False)
            list_val_data[0] = ut.Gauss_Norm_subjectWise(list_val_data[0], mu, sigma, train=False)
            list_test_data[0] = ut.Gauss_Norm_subjectWise(list_test_data[0], mu, sigma, train=False)

    if fst.flag_ttest_save ==True:
        tmp_save_dir = st.ttest_dir + '/fold_{}'.format(fold)
        ut.make_dir(dir=tmp_save_dir, flag_rm=False)
        statistic, pvalue = stats.ttest_ind(list_train_data[0][list_train_data[1] == 0][:, 0], list_train_data[0][list_train_data[1] == 1][:, 0], axis=0, equal_var=False)
        list_thresh= [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
        thresh = 0.5
        percentile = 0.1
        for i, value in enumerate(list_thresh):
            pvalue_1 = 1 * (pvalue < value)
            pvalue_2 = np.where(np.isnan(pvalue_1), 0, pvalue_1)
            pvalue_2 = pvalue_2.astype(np.float64)
            statistic_1 = statistic * (pvalue < value)
            statistic_2 = np.where(np.isnan(statistic_1), 0, statistic_1)
            ut.save_featureMap_numpy(pvalue_2, tmp_save_dir, 'binary')
            ut.save_featureMap_numpy(statistic_2, tmp_save_dir, 'p_value')
            ut.plot_heatmap_with_overlay(orig_img=list_train_data[0][0][0].squeeze(), heatmap_img=statistic_2,
                                         save_dir=tmp_save_dir + '/value_{}_{}_per_{}_th_{}_n_{}.png'.format(
                                             st.list_selected_for_train[0], st.list_selected_for_train[1],percentile, thresh, i),
                                         fig_title=str(value),
                                         thresh=thresh,
                                         percentile=percentile)

            ut.plot_heatmap_with_overlay(orig_img=list_train_data[0][0][0].squeeze(), heatmap_img=pvalue_2,
                                         save_dir=tmp_save_dir + '/binary_{}_{}_per_{}_th_{}_n_{}.png'.format(
                                             st.list_selected_for_train[0], st.list_selected_for_train[1], percentile, thresh, i),
                                         fig_title=str(value),
                                         thresh=thresh,
                                         percentile=percentile)
    if fst.flag_ttest_load == True:
        tmp_load_dir = st.ttest_dir + '/fold_{}/featuremap'.format(fold)
        t_test_binary = nib.load(tmp_load_dir + '/binary.nii.gz').get_data() # 256, 256, 256
        t_test_value = nib.load(tmp_load_dir + '/p_value.nii.gz').get_data() # 256, 256, 256

    """ make the age label different depending on the class label (NC vs MCI)"""
    # class, lbl, age, MMSE
    if fst.flag_estimate_age == True:
        for i in range(len(st.list_selected_for_train)):
            lambda_disease_factor = st.list_selected_lambdas_at_age[i]
            list_train_data[2][(list_train_data[1] == i)] = ut.estimate_biological_age(
                age=list_train_data[2][(list_train_data[1] == i)],
                MMSE=list_train_data[3][(list_train_data[1] == i)],
                lambda_disease_factor=lambda_disease_factor)

            # list_val_data[2][(list_val_data[1] == i)] = ut.estimate_biological_age(
            #     age=list_val_data[2][(list_val_data[1] == i)],
            #     MMSE=list_val_data[3][(list_val_data[1] == i)],
            #     lambda_disease_factor=lambda_disease_factor)
            #
            # list_test_data[2][(list_test_data[1] == i)] = ut.estimate_biological_age(
            #     age=list_test_data[2][(list_test_data[1] == i)],
            #     MMSE=list_test_data[3][(list_test_data[1] == i)],
            #     lambda_disease_factor=lambda_disease_factor)

    """ train loader """
    num_data = list_train_data[0].shape[0]
    train_loader = DL.convert_Dloader_3(config.batch_size, list_train_data[0], list_train_data[1], list_train_data[2], list_train_data[3],
                                        is_training=True, num_workers=1, shuffle=True)
    val_loader = DL.convert_Dloader_3(config.v_batch_size, list_val_data[0], list_val_data[1], list_val_data[2], list_val_data[3],
                                      is_training=False, num_workers=1, shuffle=False)
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2], list_test_data[3],
                                      is_training=False, num_workers=1, shuffle=False)

    del list_train_data, list_val_data, list_test_data


    print('training')
    """ epoch """
    for epoch in range(config.num_epochs):
        epoch = epoch + 1 # increase the # of the epoch
        print(" ")
        print("---------------  epoch {} ----------------".format(epoch))
        torch.cuda.empty_cache()

        """ print learning rate """
        for param_group in optimizer.param_groups:
            print('current LR : {}'.format(param_group['lr']))

        """ batch """
        for i, (datas, labels, alabels, mlabel) in enumerate(train_loader):
            # start = time.time()
            model.train()
            EMS.total_step += 1

            """ input"""
            datas = Variable(datas).cuda()
            labels = Variable(labels.long()).cuda()
            alabels = Variable(alabels.float()).cuda()

            """ forward propagation """
            dict_result = model(datas)
            output_1 = dict_result['preds']
            output_2 = dict_result['predMap']

            """ loss 1 """
            loss_list_1 = []
            ## TODO: loss depending on the disease
            if fst.flag_loss_1 == True:
                loss_list_2 = []
                for i_disease in range(len(st.list_selected_for_train)):
                    if output_1[(labels == i_disease)].nelement() != 0:
                        loss_list_2.append(criterion(output_1[(labels == i_disease)].squeeze(),
                                                     alabels[(labels == i_disease)].squeeze()) * st.list_selected_lambdas_at_loss[i_disease])
                loss_2 = sum(loss_list_2) / len(loss_list_2)
                loss_list_1.append(loss_2)
                EMS.train_aux_loss_1.append(loss_2.data.cpu().numpy())

            ## TODO: patch-wise loss
            if fst.flag_loss_2 == True:
                tmp_shape = output_2.shape
                logits = output_2.view(tmp_shape[0], tmp_shape[1], -1)
                tmp_shape = logits.shape
                loss_list_2 = []
                for i_patch in range(tmp_shape[-1]):
                    for i_disease in range(len(st.list_selected_for_train)):
                        if logits[(labels == i_disease), :, i_patch].nelement() != 0:
                            loss_list_2.append(criterion(logits[(labels == i_disease), :, i_patch].squeeze(), alabels[(labels == i_disease)].squeeze())* st.list_selected_lambdas_at_loss[i_disease])
                loss_2 = sum(loss_list_2)/len(loss_list_2)
                loss_list_1.append(loss_2)
                EMS.train_aux_loss_2.append(loss_2.data.cpu().numpy())

            loss = sum(loss_list_1)

            """ calculation of the loss through the output"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ print the train loss and tensorboard"""
            if (EMS.total_step) % 10 == 0 :
                # print('time : ', time.time() - start)
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      %(epoch, config.num_epochs, i + 1, (round(num_data / config.batch_size)), loss.data.cpu().numpy()))

            """ pyplot """
            EMS.train_loss.append(loss.data.cpu().numpy())
            EMS.train_step.append(EMS.total_step)


        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_regression_model(config, val_loader, model, criterion)
            val_loss = dict_result['Loss']
            print('Fold : %d, Epoch [%d/%d] val Loss = %f' % (fold, epoch, config.num_epochs, val_loss))

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(val_loss)
            EMS.val_step.append(EMS.total_step)

            n_stacking_loss_for_selection = 5
            if len(EMS.dict_val_metric['val_loss_queue']) > n_stacking_loss_for_selection:
                EMS.dict_val_metric['val_loss_queue'].popleft()
            EMS.dict_val_metric['val_loss_queue'].append(val_loss)
            EMS.dict_val_metric['val_mean_loss'].append(np.mean(EMS.dict_val_metric['val_loss_queue']))

            """ save model """
            for i_tmp in range(len(list_selected_EMS)):
                save_flag = ut.model_save_through_validation(fold, epoch, EMS=EMS,
                                                             selected_EMS=list_selected_EMS[i_tmp],
                                                             ES=list_ES[i_tmp],
                                                             model=model,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)



        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            """ load data """
            dict_result = ut.eval_regression_model(config, test_loader, model, criterion)
            test_loss = dict_result['Loss']

            """ pyplot """
            EMS.test_loss.append(test_loss)
            EMS.test_step.append(EMS.total_step)

            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f' % (fold, epoch, config.num_epochs, test_loss))

        """ learning rate decay"""
        EMS.LR.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        # scheduler.step(val_loss)

        """ plot the chat """
        ut.plot_training_info_1(fold, dir_pyplot, EMS)

        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model
    torch.cuda.empty_cache()

