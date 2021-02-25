import torch.utils.data as data
import numpy as np
import torch
import setting as st
import pickle
import os

# x_range = st.x_range
# y_range = st.y_range
# z_range = st.z_range
# x_size = x_range[1] - x_range[0]
# y_size = y_range[1] - y_range[0]
# z_size = z_range[1] - z_range[0]


class Dataset(data.Dataset):
    def __init__(self, Data_name, cLabel, is_training = True):
        super(Dataset, self).__init__()
        self.data = Data_name
        self.cLabel = cLabel

    def __getitem__(self,idx):
        item = torch.from_numpy(self.data[idx, ...]).float(), torch.from_numpy(self.cLabel[idx, ...])
        # item = self.transform(item)
        return item

    def __len__(self):
        return self.data.shape[0]

    # transfrom = transforms.Compose([
    #     transforms.RandomHorizontalFlip()
    # ])
def convert_Dloader(batch_size, data, label, is_training = False, num_workers = 1, shuffle = True):
        dataset = Dataset(data, label, is_training = is_training)
        # dataset = datasets.
        Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=True
                                                  )
        return Data_loader

class Dataset_2(data.Dataset):
    def __init__(self, Data_name, cLabel, aLabel, is_training = True):
        super(Dataset_2, self).__init__()
        self.data = Data_name
        self.cLabel = cLabel
        self.aLabel = aLabel

    def __getitem__(self,idx):
        item = torch.from_numpy(self.data[idx, ...]).float(), torch.from_numpy(self.cLabel[idx, ...]), torch.from_numpy(self.aLabel[idx, ...]).float()
        # item = self.transform(item)
        return item

    def __len__(self):
        return self.data.shape[0]
def convert_Dloader_2(batch_size, data, label, age, is_training = False, num_workers = 1, shuffle = True):
        dataset = Dataset_2(data, label, age, is_training = is_training)
        Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=True
                                                  )
        return Data_loader


class Dataset_3(data.Dataset):
    def __init__(self, Data_name, cLabel, aLabel, MLabel, is_training = True):
        super(Dataset_3, self).__init__()
        self.data = Data_name
        self.cLabel = cLabel
        self.aLabel = aLabel
        self.MLabel = MLabel

    def __getitem__(self,idx):
        item = torch.from_numpy(self.data[idx, ...]).float(), torch.from_numpy(self.cLabel[idx, ...]), torch.from_numpy(self.aLabel[idx, ...]).float(), torch.from_numpy(self.MLabel[idx, ...]).float()
        # item = self.transform(item)
        return item

    def __len__(self):
        return self.data.shape[0]
def convert_Dloader_3(batch_size, data, label, age, MMSE, is_training = False, num_workers = 1, shuffle = True):
        dataset = Dataset_3(data, label, age, MMSE, is_training = is_training)
        # dataset = datasets.
        Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=True
                                                  )
        return Data_loader


def concat_class_of_interest(config, fold, list_class, flag_tr_val_te ='train'):
    """
    fold = 1~10
    list_class_of_interset = [0,0,0,1,1] # should be one-hot encoded
    """
    if st.list_data_type[st.data_type_num] == 'Density':
        dtype = np.float64
    elif st.list_data_type[st.data_type_num] == 'ADNI_JSY':
        dtype = np.float32
    elif st.list_data_type[st.data_type_num] == 'ADNI_Jacob_256':
        dtype = np.uint8
    elif 'ADNI_AAL_256' in st.list_data_type[st.data_type_num]:
        dtype = np.uint8

    load_dir = st.fold_npy_dir
    fold_index = fold - 1
    list_image = []
    list_lbl = []
    list_age = []
    list_MMSE = []
    num_sample = 0
    for i_class_type in range(len(list_class)):
        if list_class[i_class_type] == 1:

            if flag_tr_val_te == 'train':
                tmp_dat_dir = st.train_fold_dir[fold_index][i_class_type]
            elif flag_tr_val_te == 'val':
                tmp_dat_dir = st.val_fold_dir[fold_index][i_class_type]
            elif flag_tr_val_te == 'test':
                tmp_dat_dir = st.test_fold_dir[fold_index][i_class_type]

            dat_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[0] + '.npy'
            list_image.append(np.memmap(filename=dat_dir, mode="r", dtype=dtype).reshape(-1, config.modality, st.x_size, st.y_size, st.z_size))
            num_sample += list_image[-1].shape[0]

            lbl_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[1] + '.npy'
            list_lbl.append(np.load(lbl_dir))

            age_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[2] + '.npy'
            list_age.append(np.load(age_dir))

            MMSE_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[3] + '.npy'
            list_MMSE.append(np.load(MMSE_dir))

    dat_image = np.vstack(list_image)
    dat_age = np.hstack(list_age)
    dat_MMSE = np.hstack(list_MMSE)

    """ make the label sequential """
    for i in range(sum(list_class)):
        if list_lbl[i][0] != i:
            list_lbl[i] = np.full_like(list_lbl[i], i, dtype=np.uint8)
    dat_lbl = np.hstack(list_lbl)

    return [dat_image, dat_lbl, dat_age, dat_MMSE]

