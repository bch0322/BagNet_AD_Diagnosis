import torch
import os
import GPUtil
from data_load import data_load as DL
import numpy as np
devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)

# torch.cuda.manual_seed(1234)
# l = torch.cuda.get_rng_state()
# torch.bernoulli(torch.full((5,5), 0.5, device='cuda'))
#
# torch.cuda.manual_seed(1234)
# l2 = torch.cuda.get_rng_state()
#
# (l==l2).all().item()

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_sample = 10
n_dim = 3
data = np.random.randint((50), size=(n_sample, n_dim))
label = np.random.randint((3), size=(n_sample,))
train_loader = DL.convert_Dloader(5, data, label, is_training=True, num_workers=0, shuffle=True)

for epoch in range(3):
    print('epoch : {}'.format(epoch))
    epoch = epoch + 1  # increase the # of the epoch
    """ batch """
    for i, (datas, labels) in enumerate(train_loader):
        print(datas)
        print(labels)
print('finish')


