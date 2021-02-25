import torch.utils.data as data
import numpy as np
import torch
import os
import setting as st
import nibabel as nib
import utils
import pickle


# x_range = st.x_range
# y_range = st.y_range
# z_range = st.z_range
# x_size = x_range[1] - x_range[0]
# y_size = y_range[1] - y_range[0]
# z_size = z_range[1] - z_range[0]

def Prepare_data_GM():
    """load the GM data 256 """
    """ GM """
    list_dir_all_modality = []

    GM_sub_list = []
    included_file_name_GM = ['gm']
    for cnt, dir in enumerate(st.dir_list):
        GM_sub_list.append(utils.search_in_whole_subdir(file_dir=st.orig_data_dir, sub_dir=dir, n_file=included_file_name_GM, n_ext='.img'))
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
                count +=1
                list_MCI.append(GM_sub_list[1][i])
        for k in range(len(GM_sub_list[4])):
            if GM_sub_list[1][i][-25:-14] == GM_sub_list[4][k][-25:-14]:
                count +=1
                list_MCI.append(GM_sub_list[1][i])
        if cur_count == count :
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

    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(list_n_subjects[i], st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.uint8))
        list_age_memallow.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
        list_MMSE_memallow.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))

    # PID_list = []
    # for i_modality in range (len(list_dir_all_modality)):
    #     for _, j_class in enumerate([0, 2]):
    #         for k in range(len(list_dir_all_modality[i_modality][j_class])):
    #             tmp_dir_file = list_dir_all_modality[i_modality][j_class][k]
    #             PID_list.append(tmp_dir_file[-24:-15])
    # print(np.unique(np.array(PID_list)).shape)

    """ save the data """
    for i_modality in range (len(list_dir_all_modality)):
        for j_class in range (len(list_dir_all_modality[i_modality])):
            for k in range(len(list_dir_all_modality[i_modality][j_class])):
                tmp_dir_file = list_dir_all_modality[i_modality][j_class][k]
                print(tmp_dir_file[-24:-14])
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, k))
                tmp_img = np.squeeze(nib.load(tmp_dir_file).get_data())[st.x_range[0] : st.x_range[1], st.y_range[0] : st.y_range[1], st.z_range[0]: st.z_range[1]]
                utils.save_numpy_to_2D_img(tmp_img, save_dir='./plot_img', file_name='/' + tmp_dir_file[-24:-14] + '_sample_class{}_{}'.format(j_class, k))
                list_image_memalloc[j_class][k, i_modality, :, :, :] = tmp_img

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
    for i_class_type in range(len(st.list_class_type)):
        list_image.append(np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=np.uint8).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
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
        train_img_dat = np.memmap(filename=save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_train_data[i_class_type].shape), dtype=np.uint8)
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
        val_img_dat = np.memmap(filename=save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_val_data[i_class_type].shape), dtype=np.uint8)
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
            mode="w+", shape=(list_test_data[i_class_type].shape), dtype=np.uint8)
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
    for i_class_type in range(len(st.list_class_type)):
        list_image.append(np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=np.uint8).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
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
    list_train_data = []  # class, subject
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
            list_train_data[i_class_type][i_sub] = utils.min_max_norm(list_train_data[i_class_type][i_sub])
        for i_sub in range(list_val_data[i_class_type].shape[0]):
            list_val_data[i_class_type][i_sub] = utils.min_max_norm(list_val_data[i_class_type][i_sub])
        for i_sub in range(list_test_data[i_class_type].shape[0]):
            list_test_data[i_class_type][i_sub] = utils.min_max_norm(list_test_data[i_class_type][i_sub])


    """ allocate the memory fold each fold  """
    # train
    print("save the train dataset")
    for i_class_type in range(len(st.list_class_type)):
        # image
        train_img_dat = np.memmap(filename=save_dir + st.train_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_train_data[i_class_type].shape), dtype=np.uint8)
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
        val_img_dat = np.memmap(filename=save_dir + st.val_fold_dir[fold_index][i_class_type] +'_'+ st.list_data_name[0] +'.npy', mode="w+",shape=(list_val_data[i_class_type].shape), dtype=np.uint8)
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
            mode="w+", shape=(list_test_data[i_class_type].shape), dtype=np.uint8)
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
