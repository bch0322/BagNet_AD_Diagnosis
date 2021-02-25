import numpy as np
import utils
import setting as st
import pickle
import os

def fold_normalization_voxelWise(config, type1_Fname, type2_Fname, load_dir, save_dir):
    """ AD, MCI, NC """
    count_1 = 0
    count_2 = 0
    for i in range(4):
        if type1_Fname[-9 - i] != '_':
            count_1 += 1
        elif type1_Fname[-9 - i] == '_':
            break

    for i in range(4):
        if type2_Fname[-9 - i] != '_':
            count_2 += 1
        elif type2_Fname[-9 - i] == '_':
            break
    tmp_1 = -8 - count_1
    tmp_2 = -8 - count_2

    for fold in range(1, config.kfold+1):
        print('fold : ' + str(fold))
        list_mu = []
        list_sigma = []
        """ load train data """
        Train_Data = np.load('%s/%s_%s_fold%d_Train_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        train_data = Train_Data['train_data']
        train_lbl = Train_Data['train_lbl']
        del Train_Data
        for i in range(train_data.shape[1]):
            train_data[:, i, :, :, :], mu, sigma = utils.Gauss_Norm_voxelWise(train_data[:, i, :, :, :], train=True)
            list_mu.append(mu)
            list_sigma.append(sigma)
        np.savez(save_dir + '/%s_%s_fold%d_Train_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),train_data=train_data, train_lbl=train_lbl)

        """ load validation data """
        Val_Data = np.load('%s/%s_%s_fold%d_Val_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("val.shape : "+str(Val_Data['val_data'].shape))
        val_data = Val_Data['val_data']
        val_lbl = Val_Data['val_lbl']
        del Val_Data
        for i in range(val_data.shape[1]):
            val_data[:, i, :, :, :]= utils.Gauss_Norm_voxelWise(val_data[:, i, :, :, :], list_mu[i], list_sigma[i], train=False)
        np.savez(save_dir + '/%s_%s_fold%d_Val_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 val_data=val_data, val_lbl=val_lbl)

        """ load test data """
        Test_Data = np.load('%s/%s_%s_fold%d_Test_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("test.shape : " + str(Test_Data['test_data'].shape))
        test_data = Test_Data['test_data']
        test_lbl = Test_Data['test_lbl']
        del Test_Data
        for i in range(val_data.shape[1]):
            test_data[:, i, :, :, :]= utils.Gauss_Norm_voxelWise(test_data[:, i, :, :, :], list_mu[i], list_sigma[i], train=False)

        np.savez(save_dir + '/%s_%s_fold%d_Test_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 test_data=test_data, test_lbl=test_lbl)


def fold_normalization_subjectWise(config, type1_Fname, type2_Fname, load_dir, save_dir):
    """ AD, MCI, NC """
    count_1 = 0
    count_2 = 0
    for i in range(4):
        if type1_Fname[-9 - i] != '_':
            count_1 += 1
        elif type1_Fname[-9 - i] == '_':
            break

    for i in range(4):
        if type2_Fname[-9 - i] != '_':
            count_2 += 1
        elif type2_Fname[-9 - i] == '_':
            break
    tmp_1 = -8 - count_1
    tmp_2 = -8 - count_2

    for fold in range(1, config.kfold+1):
        print('fold : ' + str(fold))
        list_mu = []
        list_sigma = []

        """ load train data """
        Train_Data = np.load('%s/%s_%s_fold%d_Train_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))

        train_data = Train_Data['train_data']
        train_lbl = Train_Data['train_lbl']
        del Train_Data
        for i in range(train_data.shape[1]):
            train_data[:, i, :, :, :], mu, sigma = utils.Gauss_Norm_subjectWise(train_data[:, i, :, :, :], train=True)
            list_mu.append(mu)
            list_sigma.append(sigma)

        np.savez(save_dir + '/%s_%s_fold%d_Train_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),train_data=train_data, train_lbl=train_lbl)

        """ load validation data """
        Val_Data = np.load('%s/%s_%s_fold%d_Val_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("val.shape : "+str(Val_Data['val_data'].shape))
        val_data = Val_Data['val_data']
        val_lbl = Val_Data['val_lbl']
        del Val_Data

        for i in range(val_data.shape[1]):
            val_data[:, i, :, :, :]= utils.Gauss_Norm_subjectWise(val_data[:, i, :, :, :], list_mu[i], list_sigma[i], train=False)

        np.savez(save_dir + '/%s_%s_fold%d_Val_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 val_data=val_data, val_lbl=val_lbl)

        """ load test data """
        Test_Data = np.load('%s/%s_%s_fold%d_Test_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("test.shape : " + str(Test_Data['test_data'].shape))
        test_data = Test_Data['test_data']
        test_lbl = Test_Data['test_lbl']
        del Test_Data

        for i in range(val_data.shape[1]):
            test_data[:, i, :, :, :]= utils.Gauss_Norm_subjectWise(test_data[:, i, :, :, :], list_mu[i], list_sigma[i], train=False)

        np.savez(save_dir + '/%s_%s_fold%d_Test_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 test_data=test_data, test_lbl=test_lbl)



def fold_normalization_normal(config, type1_Fname, type2_Fname, load_dir, save_dir):
    """ AD, MCI, NC """
    count_1 = 0
    count_2 = 0
    for i in range(4):
        if type1_Fname[-9 - i] != '_':
            count_1 += 1
        elif type1_Fname[-9 - i] == '_':
            break

    for i in range(4):
        if type2_Fname[-9 - i] != '_':
            count_2 += 1
        elif type2_Fname[-9 - i] == '_':
            break
    tmp_1 = -8 - count_1
    tmp_2 = -8 - count_2

    for fold in range(1, config.kfold+1):
        print('fold : ' + str(fold))

        """ load train data """
        Train_Data = np.load('%s/%s_%s_fold%d_Train_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        train_data = Train_Data['train_data']
        train_lbl = Train_Data['train_lbl']
        del Train_Data
        train_data[:, :, :, :, :] = utils.data_normalization(train_data[:, :, :, :, :])
        np.savez(save_dir + '/%s_%s_fold%d_Train_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),train_data=train_data, train_lbl=train_lbl)

        """ load validation data """
        Val_Data = np.load('%s/%s_%s_fold%d_Val_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("val.shape : "+str(Val_Data['val_data'].shape))
        val_data = Val_Data['val_data']
        val_lbl = Val_Data['val_lbl']
        del Val_Data
        val_data[:, :, :, :, :] = utils.data_normalization(val_data[:, :, :, :, :])

        np.savez(save_dir + '/%s_%s_fold%d_Val_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 val_data=val_data, val_lbl=val_lbl)

        """ load test data """
        Test_Data = np.load('%s/%s_%s_fold%d_Test_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("test.shape : " + str(Test_Data['test_data'].shape))
        test_data = Test_Data['test_data']
        test_lbl = Test_Data['test_lbl']
        del Test_Data
        test_data[:, :, :, :, :] = utils.data_normalization(test_data[:,:,:,:,:])

        np.savez(save_dir + '/%s_%s_fold%d_Test_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 test_data=test_data, test_lbl=test_lbl)

def fold_normalization_minmax(config, type1_Fname, type2_Fname, load_dir, save_dir):
    """ AD, MCI, NC """
    count_1 = 0
    count_2 = 0
    for i in range(4):
        if type1_Fname[-9 - i] != '_':
            count_1 += 1
        elif type1_Fname[-9 - i] == '_':
            break

    for i in range(4):
        if type2_Fname[-9 - i] != '_':
            count_2 += 1
        elif type2_Fname[-9 - i] == '_':
            break
    tmp_1 = -8 - count_1
    tmp_2 = -8 - count_2

    for fold in range(1, config.kfold+1):
        print('fold : ' + str(fold))

        """ load train data """
        Train_Data = np.load('%s/%s_%s_fold%d_Train_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        train_data = Train_Data['train_data']
        train_lbl = Train_Data['train_lbl']
        del Train_Data
        train_data[:, :, :, :, :] = utils.data_minmax(train_data[:, :, :, :, :])
        np.savez(save_dir + '/%s_%s_fold%d_Train_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),train_data=train_data, train_lbl=train_lbl)

        """ load validation data """
        Val_Data = np.load('%s/%s_%s_fold%d_Val_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("val.shape : "+str(Val_Data['val_data'].shape))
        val_data = Val_Data['val_data']
        val_lbl = Val_Data['val_lbl']
        del Val_Data
        val_data[:, :, :, :, :] = utils.data_minmax(val_data[:, :, :, :, :])

        np.savez(save_dir + '/%s_%s_fold%d_Val_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 val_data=val_data, val_lbl=val_lbl)

        """ load test data """
        Test_Data = np.load('%s/%s_%s_fold%d_Test_img_data.npz' % (load_dir, type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold))
        print("test.shape : " + str(Test_Data['test_data'].shape))
        test_data = Test_Data['test_data']
        test_lbl = Test_Data['test_lbl']
        del Test_Data
        test_data[:, :, :, :, :] = utils.data_minmax(test_data[:,:,:,:,:])

        np.savez(save_dir + '/%s_%s_fold%d_Test_img_data' % (type1_Fname[tmp_1:-8], type2_Fname[tmp_2:-8], fold),
                 test_data=test_data, test_lbl=test_lbl)




def Prepare_fold_data_Gaussian(config, fold, type1_Fname, type2_Fname, save_dir):
    fold_index = fold - 1
    list_mu = []
    list_sigma = []

    """ load data """
    type1_img_Data = np.memmap(filename=type1_Fname, mode="r", dtype=np.uint8).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size)
    type2_img_Data = np.memmap(filename=type2_Fname, mode="r", dtype=np.uint8).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size)
    t0_lbl = np.zeros(shape=(len(type1_img_Data), 1), dtype=np.uint8)
    t1_lbl = np.ones(shape=(len(type2_img_Data), 1), dtype=np.uint8)

    """ load the train set in the fold """
    with open(st.train_index_dir[0], 'rb') as fp:
        Type1_trIdx = pickle.load(fp)
    with open(st.train_index_dir[1], 'rb') as fp:
        Type2_trIdx = pickle.load(fp)
    train_data = np.concatenate((type1_img_Data[(Type1_trIdx[fold_index][:]), :, :, :, :], type2_img_Data[(Type2_trIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("train data shape: {}".format(train_data.shape))
    train_lbl = np.concatenate((t0_lbl[(Type1_trIdx[fold_index][:])], t1_lbl[(Type2_trIdx[fold_index][:])]), axis=0)
    print("train lbl shape: {}".format(train_lbl.shape))

    """ calculate the mean and sigma from the train set """
    for i in range(train_data.shape[1]):# 'i' is the index of modality
        train_data[:, i, :, :, :], mu, sigma = utils.Gauss_Norm_subjectWise(train_data[:, i, :, :, :], train=True)
        list_mu.append(mu)
        list_sigma.append(sigma)

    """ validation """
    with open(st.val_index_dir[0], 'rb') as fp:
        Type1_valIdx = pickle.load(fp)
    with open(st.val_index_dir[1], 'rb') as fp:
        Type2_valIdx = pickle.load(fp)
    val_data = np.concatenate((type1_img_Data[(Type1_valIdx[fold_index][:]), :, :, :, :],
                               type2_img_Data[(Type2_valIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("val data shape: {}".format(val_data.shape))
    val_lbl = np.concatenate((t0_lbl[(Type1_valIdx[fold_index][:])], t1_lbl[(Type2_valIdx[fold_index][:])]), axis=0)
    print("val lbl shape: {}".format(val_lbl.shape))

    for i in range(val_data.shape[1]):# 'i' is the index of modality
        val_data[:, i, :, :, :] = utils.Gauss_Norm_subjectWise(val_data[:, i, :, :, :],list_mu[i], list_sigma[i], train=False)

    """ test """
    with open(st.test_index_dir[0], 'rb') as fp:
        Type1_teIdx = pickle.load(fp)
    with open(st.test_index_dir[1], 'rb') as fp:
        Type2_teIdx = pickle.load(fp)
    test_data = np.concatenate(
        (type1_img_Data[(Type1_teIdx[fold_index][:]), :, :, :, :],
         type2_img_Data[(Type2_teIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("test data shape: {}".format(test_data.shape))
    test_lbl = np.concatenate((t0_lbl[(Type1_teIdx[fold_index][:])], t1_lbl[(Type2_teIdx[fold_index][:])]), axis=0)
    print("test lbl shape: {}".format(test_lbl.shape))

    for i in range(test_data.shape[1]):# 'i' is the index of modality
        test_data[:, i, :, :, :] = utils.Gauss_Norm_subjectWise(test_data[:, i, :, :, :],list_mu[i], list_sigma[i], train=False)


    """ allocate the memory fold each fold  """
    # train

    train_img_dat = np.memmap(filename=save_dir + '/train_dat_fold_{}.npy'.format(fold), mode="w+",
                              shape=(train_data.shape), dtype=np.uint8)
    train_img_dat[:] = train_data[:]
    del train_img_dat

    train_lbl_dir = save_dir + '/train_lbl_fold_{}.npy'.format(fold)
    np.save(train_lbl_dir, train_lbl)

    # val
    val_img_dat = np.memmap(filename=save_dir + '/val_dat_fold_{}.npy'.format(fold), mode="w+",
                            shape=(val_data.shape), dtype=np.uint8)
    val_img_dat[:] = val_data[:]
    val_lbl_dir = save_dir + '/val_lbl_fold_{}.npy'.format(fold)
    np.save(val_lbl_dir, val_lbl)

    # test
    test_img_dat = np.memmap(filename=save_dir + '/test_dat_fold_{}.npy'.format(fold), mode="w+",
                             shape=(test_data.shape), dtype=np.uint8)
    test_img_dat[:] = test_data[:]
    test_lbl_dir = save_dir + '/test_lbl_fold_{}.npy'.format(fold)
    np.save(test_lbl_dir, test_lbl)


def Prepare_fold_data_voxelWise(config, fold, type1_Fname, type2_Fname, save_dir):
    fold_index = fold - 1
    list_mu = []
    list_sigma = []

    """ load data """
    type1_img_Data = np.memmap(filename=type1_Fname, mode="r", dtype=np.uint8).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size)
    type2_img_Data = np.memmap(filename=type2_Fname, mode="r", dtype=np.uint8).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size)
    t0_lbl = np.zeros(shape=(len(type1_img_Data), 1), dtype=np.uint8)
    t1_lbl = np.ones(shape=(len(type2_img_Data), 1), dtype=np.uint8)

    """ load the train set in the fold """
    with open(st.train_index_dir[0], 'rb') as fp:
        Type1_trIdx = pickle.load(fp)
    with open(st.train_index_dir[1], 'rb') as fp:
        Type2_trIdx = pickle.load(fp)
    train_data = np.concatenate((type1_img_Data[(Type1_trIdx[fold_index][:]), :, :, :, :], type2_img_Data[(Type2_trIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("train data shape: {}".format(train_data.shape))
    train_lbl = np.concatenate((t0_lbl[(Type1_trIdx[fold_index][:])], t1_lbl[(Type2_trIdx[fold_index][:])]), axis=0)
    print("train lbl shape: {}".format(train_lbl.shape))

    """ calculate the mean and sigma from the train set """
    for i in range(train_data.shape[1]):# 'i' is the index of modality
        train_data[:, i, :, :, :], mu, sigma = utils.Gauss_Norm_voxelWise(train_data[:, i, :, :, :], train=True)
        list_mu.append(mu)
        list_sigma.append(sigma)

    """ validation """
    with open(st.val_index_dir[0], 'rb') as fp:
        Type1_valIdx = pickle.load(fp)
    with open(st.val_index_dir[1], 'rb') as fp:
        Type2_valIdx = pickle.load(fp)
    val_data = np.concatenate((type1_img_Data[(Type1_valIdx[fold_index][:]), :, :, :, :],
                               type2_img_Data[(Type2_valIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("val data shape: {}".format(val_data.shape))
    val_lbl = np.concatenate((t0_lbl[(Type1_valIdx[fold_index][:])], t1_lbl[(Type2_valIdx[fold_index][:])]), axis=0)
    print("val lbl shape: {}".format(val_lbl.shape))

    for i in range(val_data.shape[1]):# 'i' is the index of modality
        val_data[:, i, :, :, :] = utils.Gauss_Norm_voxelWise(val_data[:, i, :, :, :],list_mu[i], list_sigma[i], train=False)

    """ test """
    with open(st.test_index_dir[0], 'rb') as fp:
        Type1_teIdx = pickle.load(fp)
    with open(st.test_index_dir[1], 'rb') as fp:
        Type2_teIdx = pickle.load(fp)
    test_data = np.concatenate(
        (type1_img_Data[(Type1_teIdx[fold_index][:]), :, :, :, :],
         type2_img_Data[(Type2_teIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("test data shape: {}".format(test_data.shape))
    test_lbl = np.concatenate((t0_lbl[(Type1_teIdx[fold_index][:])], t1_lbl[(Type2_teIdx[fold_index][:])]), axis=0)
    print("test lbl shape: {}".format(test_lbl.shape))

    for i in range(test_data.shape[1]):# 'i' is the index of modality
        test_data[:, i, :, :, :] = utils.Gauss_Norm_voxelWise(test_data[:, i, :, :, :],list_mu[i], list_sigma[i], train=False)


    """ allocate the memory fold each fold  """
    # train
    train_img_dat = np.memmap(filename=save_dir + '/train_dat_fold_{}.npy'.format(fold), mode="w+",
                              shape=(train_data.shape), dtype=np.uint8)
    train_img_dat[:] = train_data[:]
    del train_img_dat

    train_lbl_dir = save_dir + '/train_lbl_fold_{}.npy'.format(fold)
    np.save(train_lbl_dir, train_lbl)

    # val
    val_img_dat = np.memmap(filename=save_dir + '/val_dat_fold_{}.npy'.format(fold), mode="w+",
                            shape=(val_data.shape), dtype=np.uint8)
    val_img_dat[:] = val_data[:]
    val_lbl_dir = save_dir + '/val_lbl_fold_{}.npy'.format(fold)
    np.save(val_lbl_dir, val_lbl)

    # test
    test_img_dat = np.memmap(filename=save_dir + '/test_dat_fold_{}.npy'.format(fold), mode="w+",
                             shape=(test_data.shape), dtype=np.uint8)
    test_img_dat[:] = test_data[:]
    test_lbl_dir = save_dir + '/test_lbl_fold_{}.npy'.format(fold)
    np.save(test_lbl_dir, test_lbl)








def Prepare_normalized_fold_GM_minmax(config, fold, type1_Fname, type2_Fname, save_dir):
    fold_index = fold - 1
    list_mu = []
    list_sigma = []

    ###################################
    """ load data """
    type1_img_Data = np.memmap(filename=type1_Fname, mode="r", dtype=np.uint8).reshape(-1, config.modality,
                                                                                       st.x_size, st.y_size,
                                                                                       st.z_size)
    type2_img_Data = np.memmap(filename=type2_Fname, mode="r", dtype=np.uint8).reshape(-1, config.modality,
                                                                                       st.x_size, st.y_size,
                                                                                       st.z_size)
    t0_lbl = np.zeros(shape=(len(type1_img_Data), 1), dtype=np.uint8)
    t1_lbl = np.ones(shape=(len(type2_img_Data), 1), dtype=np.uint8)

    #####################################
    """ load the fold list for train """
    with open(st.train_index_dir[0], 'rb') as fp:
        Type1_trIdx = pickle.load(fp)
    with open(st.train_index_dir[1], 'rb') as fp:
        Type2_trIdx = pickle.load(fp)

    train_data = np.concatenate((type1_img_Data[(Type1_trIdx[fold_index][:]), :, :, :, :],
                                 type2_img_Data[(Type2_trIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("train data shape: {}".format(train_data.shape))
    train_lbl = np.concatenate((t0_lbl[(Type1_trIdx[fold_index][:])], t1_lbl[(Type2_trIdx[fold_index][:])]), axis=0)
    print("train lbl shape: {}".format(train_lbl.shape))

    for i in range(train_data.shape[1]):
        train_data[:, i, :,:,:] = utils.data_minmax(train_data[:, i, :, :, :])
        # train_data[:, i, :, :, :], mu, sigma = utils.Gauss_Norm_voxelWise(train_data[:, i, :, :, :], train=True)
        # list_mu.append(mu)
        # list_sigma.append(sigma)

    ###############################################################################
    """ load the fold list for validation """
    with open(st.val_index_dir[0], 'rb') as fp:
        Type1_valIdx = pickle.load(fp)
    with open(st.val_index_dir[1], 'rb') as fp:
        Type2_valIdx = pickle.load(fp)
    val_data = np.concatenate((type1_img_Data[(Type1_valIdx[fold_index][:]), :, :, :, :],
                               type2_img_Data[(Type2_valIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("val data shape: {}".format(val_data.shape))
    val_lbl = np.concatenate((t0_lbl[(Type1_valIdx[fold_index][:])], t1_lbl[(Type2_valIdx[fold_index][:])]), axis=0)
    print("val lbl shape: {}".format(val_lbl.shape))

    for i in range(val_data.shape[1]):
        val_data[:, i, :, :, :] = utils.Gauss_Norm_voxelWise(val_data[:, i, :, :, :], list_mu[i], list_sigma[i],
                                                             train=False)
    ###############################################################################
    """ load the fold list for test """
    with open(st.test_index_dir[0], 'rb') as fp:
        Type1_teIdx = pickle.load(fp)
    with open(st.test_index_dir[1], 'rb') as fp:
        Type2_teIdx = pickle.load(fp)
    test_data = np.concatenate(
        (type1_img_Data[(Type1_teIdx[fold_index][:]), :, :, :, :],
         type2_img_Data[(Type2_teIdx[fold_index])[:], :, :, :, :]), axis=0)
    print("test data shape: {}".format(test_data.shape))
    test_lbl = np.concatenate((t0_lbl[(Type1_teIdx[fold_index][:])], t1_lbl[(Type2_teIdx[fold_index][:])]), axis=0)
    print("test lbl shape: {}".format(test_lbl.shape))

    for i in range(test_data.shape[1]):
        test_data[:, i, :, :, :] = utils.Gauss_Norm_voxelWise(test_data[:, i, :, :, :], list_mu[i], list_sigma[i], train=False)

    ## saving
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    """ allocate the memory fold each fold  """
    # train
    train_img_dat = np.memmap(filename=save_dir + '/train_dat_fold_{}.npy'.format(fold), mode="w+", shape=(train_data.shape), dtype=np.uint8)
    train_img_dat[:] = train_data[:]
    del train_img_dat
    train_lbl_dir = save_dir + '/train_lbl_fold_{}.npy'.format(fold)
    np.save(train_lbl_dir, train_lbl)

    # val
    val_img_dat = np.memmap(filename=save_dir + '/val_dat_fold_{}.npy'.format(fold), mode="w+", shape=(val_data.shape), dtype=np.uint8)
    val_img_dat[:] = val_data[:]
    del val_img_dat
    val_lbl_dir = save_dir + '/val_lbl_fold_{}.npy'.format(fold)
    np.save(val_lbl_dir, val_lbl)

    # test
    test_img_dat = np.memmap(filename=save_dir + '/test_dat_fold_{}.npy'.format(fold), mode="w+", shape=(test_data.shape), dtype=np.uint8)
    test_img_dat[:] = test_data[:]
    del test_img_dat
    test_lbl_dir = save_dir + '/test_lbl_fold_{}.npy'.format(fold)
    np.save(test_lbl_dir, test_lbl)
