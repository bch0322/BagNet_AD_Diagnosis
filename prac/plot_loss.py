import numpy as np
import utils as ut

step = np.arange(10)
loss_a = np.random.rand(10)
loss_b = np.random.rand(10)
loss_c = np.random.rand(10)
loss_d = np.random.rand(10)

list_y = []
list_y_name = []
list_y.append(loss_a)
list_y_name.append('train_acc_2')
list_y.append(loss_b)
list_y_name.append('train_acc_5')
list_y.append(loss_c)
list_y_name.append('train_acc_4')
list_y.append(loss_d)
list_y_name.append('train_acc_7')
ut.plot_list_v2(step, list_y, title='train acc', n_xlabel='step', n_ylabel=list_y_name,
                save_dir='./', file_name='/fold_{0}_train_acc'.format(1), flag='minmax')