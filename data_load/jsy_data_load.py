import pickle
import numpy as np
import setting as st
import pandas as pd
import utils as ut
import os

def Prepare_data_1():
    """ load data jsy processed """
    dat_dir = st.orig_data_dir + '/adni_dat.npy'
    cls_dir = st.orig_data_dir + '/adni_cls.npy'
    age_dir = st.orig_data_dir + '/adni_age.npy'
    id_dir = st.orig_data_dir + '/adni_id.npy'

    adni_dat = np.load(dat_dir, mmap_mode='r')
    adni_cls = np.load(cls_dir, mmap_mode='r')
    adni_age = np.load(age_dir, mmap_mode='r')
    adni_id = np.load(id_dir, mmap_mode='r')

    # t_adni_cls = adni_cls

    """ allocation memory """
    list_image_memalloc = []
    list_age_memallow = []
    list_MMSE_memallow = []


    """ the # of the subject depending on the disease label """
    unique, counts = np.unique(adni_cls, return_counts=True)

    n_NC_subjects = counts[0]
    n_MCI_subjects = counts[1]
    n_AD_subjects = counts[2]
    list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects]
    # n_sMCI_subjects = list_final_label.count(1)
    # n_pMCI_subjects = list_final_label.count(2)
    # list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects, n_sMCI_subjects, n_pMCI_subjects]

    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(list_n_subjects[i], st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memallow.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
        list_MMSE_memallow.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
    #
    """ save the data """
    count_NC = 0
    count_MCI = 0
    count_AD = 0
    count_total_samples = 0
    for j in range(adni_dat.shape[0]):
        print(f'{j}th subject.')
        count_total_samples +=1
        if adni_cls[j] == 0:
            list_image_memalloc[0][count_NC, 0, :, :, :]= np.squeeze(adni_dat[j])
            list_age_memallow[0][count_NC] = np.squeeze(adni_age[j])
            count_NC += 1

        elif adni_cls[j] == 1:
            list_image_memalloc[1][count_MCI, 0, :, :, :]= np.squeeze(adni_dat[j])
            list_age_memallow[1][count_MCI] = np.squeeze(adni_age[j])
            count_MCI += 1

        elif adni_cls[j] == 2:
            list_image_memalloc[2][count_AD, 0, :, :, :]= np.squeeze(adni_dat[j])
            list_age_memallow[2][count_AD] = np.squeeze(adni_age[j])
            count_AD += 1

    print("count nc : " + str(count_NC)) # 284
    print("count mci : " + str(count_MCI)) # 374
    print("count ad : " + str(count_AD))  # 329


def Prepare_fold_data_1(config, fold):
    save_dir = st.fold_npy_dir
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    fold_index = fold - 1
    """ load data """
    list_image = []
    list_age = []
    list_MMSE = []
    list_lbl = []
    if st.list_data_type[st.data_type_num] == 'ADNI_JSY':
        for i_class_type in range(len(st.list_class_type)):
            list_image.append(np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=np.float32).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
            list_age.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r", dtype=np.float32))
            list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r", dtype=np.float32))
            list_lbl.append(np.full(shape=(len(list_image[i_class_type])), fill_value=i_class_type, dtype=np.uint8))

    """ load the fold list for train """
    list_trIdx = []
    list_valIdx = []
    list_teIdx = []
    for i_class_type in range(len(st.list_class_type)):
        with open(st.train_index_dir[i_class_type], 'rb') as fp:
            list_trIdx.append(pickle.load(fp))
        with open(st.val_index_dir[i_class_type], 'rb') as fp:
            list_valIdx.append(pickle.load(fp))
        with open(st.test_index_dir[i_class_type], 'rb') as fp:
            list_teIdx.append(pickle.load(fp))

    """ apply fold index using corresponding fold index """
    list_train_data = []
    list_train_lbl = []
    list_train_age = []
    list_train_MMSE = []

    list_val_data = []
    list_val_lbl = []
    list_val_age = []
    list_val_MMSE = []

    list_test_data = []
    list_test_lbl = []
    list_test_age = []
    list_test_MMSE = []
    for i_class_type in range(len(st.list_class_type)):
        print("disease_class : {}".format(i_class_type))
        list_train_data.append(list_image[i_class_type][(list_trIdx[i_class_type][fold_index][:]), :, :, :, :])
        list_train_lbl.append(list_lbl[i_class_type][(list_trIdx[i_class_type][fold_index][:])])
        list_train_age.append(list_age[i_class_type][(list_trIdx[i_class_type][fold_index][:])])
        list_train_MMSE.append(list_MMSE[i_class_type][(list_trIdx[i_class_type][fold_index][:])])

        list_val_data.append(list_image[i_class_type][(list_valIdx[i_class_type][fold_index][:]), :, :, :, :])
        list_val_lbl.append(list_lbl[i_class_type][(list_valIdx[i_class_type][fold_index][:])])
        list_val_age.append(list_age[i_class_type][(list_valIdx[i_class_type][fold_index][:])])
        list_val_MMSE.append(list_MMSE[i_class_type][(list_valIdx[i_class_type][fold_index][:])])

        list_test_data.append(list_image[i_class_type][(list_teIdx[i_class_type][fold_index][:]), :, :, :, :])
        list_test_lbl.append(list_lbl[i_class_type][(list_teIdx[i_class_type][fold_index][:])])
        list_test_age.append(list_age[i_class_type][(list_teIdx[i_class_type][fold_index][:])])
        list_test_MMSE.append(list_MMSE[i_class_type][(list_teIdx[i_class_type][fold_index][:])])

    """ allocate the memory fold each fold  """
    # train
    print("save the train dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        train_img_dat = np.memmap(filename=save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_train_data[i_class_type].shape), dtype=np.float32)
        train_img_dat[:] = list_train_data[i_class_type]

        # lbl
        train_lbl_dir = save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[1] +'.npy'
        np.save(train_lbl_dir, list_train_lbl[i_class_type])

        # age
        train_age_dir = save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[2] +'.npy'
        np.save(train_age_dir, list_train_age[i_class_type])

        # MMSE
        train_MMSE_dir = save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[3] +'.npy'
        np.save(train_MMSE_dir, list_train_MMSE[i_class_type])


    # val
    print("save the val dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        val_img_dat = np.memmap(filename=save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_val_data[i_class_type].shape), dtype=np.float32)
        val_img_dat[:] = list_val_data[i_class_type]

        # lbl
        val_lbl_dir = save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[1] +'.npy'
        np.save(val_lbl_dir, list_val_lbl[i_class_type])

        # age
        val_age_dir = save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[2] +'.npy'
        np.save(val_age_dir, list_val_age[i_class_type])

        # MMSE
        val_MMSE_dir = save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[3] +'.npy'
        np.save(val_MMSE_dir, list_val_MMSE[i_class_type])

    # test
    print("save the test dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        test_img_dat = np.memmap(
            filename=save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy',
            mode="w+", shape=(list_test_data[i_class_type].shape), dtype=np.float32)
        test_img_dat[:] = list_test_data[i_class_type]

        # lbl
        test_lbl_dir = save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[1] +'.npy'
        np.save(test_lbl_dir, list_test_lbl[i_class_type])

        # age
        test_age_dir = save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[2] +'.npy'
        np.save(test_age_dir, list_test_age[i_class_type])

        # MMSE
        test_MMSE_dir = save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[3] +'.npy'
        np.save(test_MMSE_dir, list_test_MMSE[i_class_type])


def Prepare_fold_data_2(config, fold):
    save_dir = st.fold_npy_dir
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    fold_index = fold - 1

    """ load data """
    list_image = []
    list_age = []
    list_MMSE = []
    list_lbl = []
    if st.list_data_type[st.data_type_num] == 'ADNI_JSY':
        for i_class_type in range(len(st.list_class_type)):
            list_image.append(np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r+", dtype=np.float32).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
            list_age.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r+", dtype=np.float32))
            list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r+", dtype=np.float32))
            list_lbl.append(np.full(shape=(len(list_image[i_class_type])), fill_value=i_class_type, dtype=np.uint8))



    """ load the fold list for train """
    list_trIdx = []
    list_valIdx = []
    list_teIdx = []
    for i_class_type in range(len(st.list_class_type)):
        with open(st.train_index_dir[i_class_type], 'rb') as fp:
            list_trIdx.append(pickle.load(fp))
        with open(st.val_index_dir[i_class_type], 'rb') as fp:
            list_valIdx.append(pickle.load(fp))
        with open(st.test_index_dir[i_class_type], 'rb') as fp:
            list_teIdx.append(pickle.load(fp))

    """ apply fold index using corresponding fold index """
    list_train_data = []
    list_train_lbl = []
    list_train_age = []
    list_train_MMSE = []

    list_val_data = []
    list_val_lbl = []
    list_val_age = []
    list_val_MMSE = []

    list_test_data = []
    list_test_lbl = []
    list_test_age = []
    list_test_MMSE = []
    for i_class_type in range(len(st.list_class_type)):
        print("disease_class : {}".format(i_class_type))
        list_train_data.append(list_image[i_class_type][(list_trIdx[i_class_type][fold_index][:]), :, :, :, :])
        list_train_lbl.append(list_lbl[i_class_type][(list_trIdx[i_class_type][fold_index][:])])
        list_train_age.append(list_age[i_class_type][(list_trIdx[i_class_type][fold_index][:])])
        list_train_MMSE.append(list_MMSE[i_class_type][(list_trIdx[i_class_type][fold_index][:])])

        list_val_data.append(list_image[i_class_type][(list_valIdx[i_class_type][fold_index][:]), :, :, :, :])
        list_val_lbl.append(list_lbl[i_class_type][(list_valIdx[i_class_type][fold_index][:])])
        list_val_age.append(list_age[i_class_type][(list_valIdx[i_class_type][fold_index][:])])
        list_val_MMSE.append(list_MMSE[i_class_type][(list_valIdx[i_class_type][fold_index][:])])

        list_test_data.append(list_image[i_class_type][(list_teIdx[i_class_type][fold_index][:]), :, :, :, :])
        list_test_lbl.append(list_lbl[i_class_type][(list_teIdx[i_class_type][fold_index][:])])
        list_test_age.append(list_age[i_class_type][(list_teIdx[i_class_type][fold_index][:])])
        list_test_MMSE.append(list_MMSE[i_class_type][(list_teIdx[i_class_type][fold_index][:])])


    """ normalization """
    for i_class_type in range(len(st.list_class_type)):
        for i_sub in range(list_train_data[i_class_type].shape[0]):
            list_train_data[i_class_type][i_sub] = ut.min_max_norm(list_train_data[i_class_type][i_sub])
        for i_sub in range(list_val_data[i_class_type].shape[0]):
            list_val_data[i_class_type][i_sub] = ut.min_max_norm(list_val_data[i_class_type][i_sub])
        for i_sub in range(list_test_data[i_class_type].shape[0]):
            list_test_data[i_class_type][i_sub] = ut.min_max_norm(list_test_data[i_class_type][i_sub])

    """ allocate the memory fold each fold  """
    print("save the train dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        train_img_dat = np.memmap(filename=save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_train_data[i_class_type].shape), dtype=np.float32)
        train_img_dat[:] = list_train_data[i_class_type]

        # lbl
        train_lbl_dir = save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[1] +'.npy'
        np.save(train_lbl_dir, list_train_lbl[i_class_type])

        # age
        train_age_dir = save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[2] +'.npy'
        np.save(train_age_dir, list_train_age[i_class_type])

        # MMSE
        train_MMSE_dir = save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[3] +'.npy'
        np.save(train_MMSE_dir, list_train_MMSE[i_class_type])


    # val
    print("save the val dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        val_img_dat = np.memmap(filename=save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_val_data[i_class_type].shape), dtype=np.float32)
        val_img_dat[:] = list_val_data[i_class_type]

        # lbl
        val_lbl_dir = save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[1] +'.npy'
        np.save(val_lbl_dir, list_val_lbl[i_class_type])

        # age
        val_age_dir = save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[2] +'.npy'
        np.save(val_age_dir, list_val_age[i_class_type])

        # MMSE
        val_MMSE_dir = save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[3] +'.npy'
        np.save(val_MMSE_dir, list_val_MMSE[i_class_type])

    # test
    print("save the test dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        test_img_dat = np.memmap(
            filename=save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy',
            mode="w+", shape=(list_test_data[i_class_type].shape), dtype=np.float32)
        test_img_dat[:] = list_test_data[i_class_type]

        # lbl
        test_lbl_dir = save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[1] +'.npy'
        np.save(test_lbl_dir, list_test_lbl[i_class_type])

        # age
        test_age_dir = save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[2] +'.npy'
        np.save(test_age_dir, list_test_age[i_class_type])

        # MMSE
        test_MMSE_dir = save_dir + st.test_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[3] +'.npy'
        np.save(test_MMSE_dir, list_test_MMSE[i_class_type])
