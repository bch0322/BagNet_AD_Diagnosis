import os
import sys
import utils as ut
from torch.autograd import Variable
import torch
import torch.nn as nn
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import numpy as np

def train(config, fold, model, model_1, optimizer, scheduler, list_dir_save_model, dir_pyplot, Validation=True, Test_flag = True):

    """ loss """
    # criterion_cls = nn.CrossEntropyLoss()
    criterion_cls = ut.FocalLoss(gamma=st.focal_gamma, alpha=st.focal_alpha)
    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=st.early_stopping_patience, verbose=True))

    """ load data """
    list_train_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_train, flag_tr_val_te='train')
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
        for i, (datas, labels, alabel, mlabel) in enumerate(train_loader):
            # start = time.time()
            model.train()
            EMS.total_step += 1


            """ input"""
            datas = Variable(datas).cuda()
            labels = Variable(labels.long()).cuda()

            """ forward propagation """
            dict_result = model_1(datas)
            final_evidence = dict_result['final_evidence']
            featureMaps = dict_result['featureMaps']

            dict_result = model(datas, final_evidence, featureMaps)
            output_1 = dict_result['logits']
            output_2 = dict_result['Aux_logits']
            output_3 = dict_result['logitMap']

            """ loss 1 """
            loss_list_1 = []
            if fst.flag_loss_1 == True:
                loss_2 = criterion_cls(output_1, labels)
                loss_list_1.append(loss_2)
                EMS.train_aux_loss_1.append(loss_2.data.cpu().numpy())

            ## TODO : aux_loss
            if fst.flag_loss_2 == True:
                loss_2 = criterion_cls(output_2, labels)
                loss_list_1.append(loss_2)
                EMS.train_aux_loss_2.append(loss_2.data.cpu().numpy())

            ## TODO : patch-level loss
            if fst.flag_loss_3 == True:
                tmp_shape = output_3.shape
                logits = output_3.view(tmp_shape[0], tmp_shape[1], -1)
                tmp_shape = logits.shape
                loss_list_2 = []
                for i_patch in range(tmp_shape[-1]):
                    loss_list_2.append(criterion_cls(logits[:, :, i_patch], labels))
                loss_2 = sum(loss_list_2)/ len(loss_list_2)
                loss_list_1.append(loss_2)
                EMS.train_aux_loss_3.append(loss_2.data.cpu().numpy())

            ## TODO : depending on the disese
            if fst.flag_loss_4 == True:
                loss_list_2 = []
                for i_disease in range(len(st.list_selected_for_train)):
                    if output_1[(labels == i_disease)].nelement() != 0:
                        loss_list_2.append(criterion_cls(output_1[(labels == i_disease)].squeeze(), alabel[(labels == i_disease)].squeeze()) * st.list_selected_lambdas_at_loss[i_disease])
                loss_2 = sum(loss_list_2)/len(loss_list_2)
                loss_list_1.append(loss_2)
                EMS.train_aux_loss_4.append(loss_2.data.cpu().numpy())

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


        """ learning rate decay"""
        EMS.LR.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_classification_using_pretrained(config, val_loader, model, model_1, criterion_cls)
            val_loss = dict_result['Loss']
            acc = dict_result['Acc']
            auc = dict_result['AUC']
            print('Fold : %d, Epoch [%d/%d] val Loss = %f val Acc = %f' % (fold, epoch, config.num_epochs, val_loss, acc))

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(val_loss)
            EMS.dict_val_metric['val_acc'].append(acc)
            EMS.dict_val_metric['val_auc'].append(auc)
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
                                                             ES = list_ES[i_tmp],
                                                             model=model,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)


        """ ------------ """
        """ train dataset """
        """ ------------ """
        if fst.flag_print_trainAcc== True:
            print("------------------  test _ train dataset  --------------------------")
            """ eval """
            dict_result = ut.eval_classification_using_pretrained(config, train_loader, model, model_1, criterion_cls)
            acc = dict_result['Acc']
            train_loss = dict_result['Loss']

            """ pyplot """
            EMS.train_acc.append(acc)

            """ print out """

            print('number of train samples : {}'.format(len(train_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] train Loss = %f train Acc = %f' % (fold, epoch, config.num_epochs, train_loss, acc))


        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            """ load data """
            dict_result = ut.eval_classification_using_pretrained(config, test_loader, model, model_1, criterion_cls)
            acc = dict_result['Acc']
            test_loss = dict_result['Loss']

            """ pyplot """
            # test_acc_list.append(acc)
            EMS.test_acc.append(acc)
            EMS.test_loss.append(test_loss)
            EMS.test_step.append(EMS.total_step)

            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f test Acc = %f' % (fold, epoch, config.num_epochs, test_loss, acc))



        """ plot the chat"""

        """ train loss """
        train_loss_plot_list = []
        train_loss_plot_list.append(EMS.train_loss)
        ut.plot_list_v1(EMS.train_step, train_loss_plot_list, title='train loss', n_xlabel='step', n_ylabel='train loss',
                        save_dir=dir_pyplot, file_name='/fold_{0}_train_loss'.format(fold), flag='minmax')

        """ train aux loss """
        if len(EMS.train_aux_loss_1) == len(EMS.train_step):
            train_aux_loss_plot_list = []
            train_aux_loss_plot_list.append(EMS.train_aux_loss_1)
            ut.plot_list_v1(EMS.train_step, train_aux_loss_plot_list, title='train aux loss', n_xlabel='step', n_ylabel='train aux loss',
                            save_dir=dir_pyplot, file_name='/fold_{0}_train_aux_loss'.format(fold), flag='minmax')

        """ val acc """
        val_acc_plot_list = []
        val_acc_plot_list.append(EMS.dict_val_metric['val_acc'])
        ut.plot_list_v1(EMS.val_step, val_acc_plot_list, title='val_acc', n_xlabel='step', n_ylabel='val acc',
                        save_dir=dir_pyplot, file_name='/fold_{0}_val_acc'.format(fold), flag='minmax')

        """ val loss """
        val_loss_plot_list = []
        val_loss_plot_list.append(EMS.dict_val_metric['val_loss'])
        ut.plot_list_v1(EMS.val_step, val_loss_plot_list, title='val_loss', n_xlabel='step', n_ylabel='val loss',
                        save_dir=dir_pyplot, file_name='/fold_{0}_val_loss'.format(fold), flag='minmax')

        """ test acc """
        test_acc_plot_list = []
        test_acc_plot_list.append(EMS.test_acc)
        ut.plot_list_v1(EMS.test_step, test_acc_plot_list, title='test acc', n_xlabel='step', n_ylabel='test acc',
                        save_dir=dir_pyplot, file_name='/fold_{0}_test'.format(fold), flag='minmax')

        LR_plot_list = []
        LR_plot_list.append(EMS.LR)
        ut.plot_list_v1(EMS.val_step, LR_plot_list, title='Learning rate', n_xlabel='step', n_ylabel='Learning rate',
                        save_dir=dir_pyplot, file_name='/fold_{0}_Learning_rate'.format(fold), flag='minmax')


        """ train acc """
        if fst.flag_print_trainAcc==True:
            train_acc_plot_list = []
            train_acc_plot_list.append(EMS.train_acc)
            ut.plot_list_v1(EMS.train_step, train_acc_plot_list, title='train acc', n_xlabel='step', n_ylabel='train acc',
                            save_dir=dir_pyplot, file_name='/fold_{0}_train_acc'.format(fold), flag='minmax')
            del train_acc_plot_list

        # del val_loss_plot_list, val_acc_plot_list, train_loss_plot_list, test_acc_plot_list
        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model
    torch.cuda.empty_cache()

