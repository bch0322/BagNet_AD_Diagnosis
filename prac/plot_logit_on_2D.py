import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


tmp_dir_file_0 = './logit_0.nii.gz'
tmp_dir_file_1 = './logit_1.nii.gz'
logit_0_img = nib.load(tmp_dir_file_0).get_fdata()
logit_1_img = nib.load(tmp_dir_file_1).get_fdata()

x =logit_0_img
y = logit_1_img
print("x : {}".format(x.reshape(-1).sum()))
print("y : {}".format(y.reshape(-1).sum()))
## TODO: scatter logit
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, s=1)
min = int(min([x.reshape(-1).min(), y.reshape(-1).min()]) * 1.2)
max = int(max([x.reshape(-1).max(), y.reshape(-1).max()]) * 1.2)
ax.plot(range(min, max), range(min, max))
ax.set_xlim([min, max])
ax.set_ylim([min, max])
ax.grid(True)
plt.axis('square')
plt.savefig('test_{}'.format(1))
print('!!')
