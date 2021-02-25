import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import os
import sys
file_dir_orig = '../input.nii.gz'
file_dir_heatmap = '../final_evidence_map.nii.gz'

# file_dir_orig = './input_fold_1_sample_1_gt_MCI.nii.gz'
# file_dir_heatmap = './heatmap_1_fold_1_sample_1_gt_MCI.nii.gz'
# file_dir_heatmap = './heatmap_2_fold_1_sample_1_gt_MCI.nii.gz'

orig_img = nib.load(file_dir_orig).get_data()
heatmap_img = nib.load(file_dir_heatmap).get_data()
print(heatmap_img.shape)

shape = heatmap_img.shape
list_interval = []
for j in range(3):
    tmp_list = []
    for i in np.arange(20, 81, 4):
        tmp_list.append(int(np.percentile(np.arange(0, shape[j]), i)))
    list_interval.append(np.hstack(tmp_list))

axis_type = ['Sagittal', 'Coronal', 'Axial']

fig = plt.figure(figsize=(list_interval[0].shape[0] * 3, len(axis_type) * 3))
plt.rcParams.update({'font.size': 20})
fig.suptitle('HeatMap', fontsize=30)

heights = [1] * len(axis_type)
widths = [10] * (list_interval[0].shape[0])
widths.append(10)
gs = gridspec.GridSpec(nrows=len(heights),  # row
                       ncols=len(widths),
                       height_ratios=heights,
                       width_ratios=widths,
                       hspace=0.0,
                       wspace=0.0,
                       )

cmap_orig = plt.get_cmap('Greys')

# cmap_heatmap = plt.get_cmap('Reds')
cmap_heatmap = plt.get_cmap('coolwarm')
# cmap_heatmap = plt.get_cmap('bwr')

# for orig
orig_vmax = np.percentile(orig_img, 99)
orig_vmin = np.percentile(orig_img, 1)
print(orig_vmin, orig_vmax)

vmax = np.percentile(heatmap_img, 99)
vmin = np.percentile(heatmap_img, 1)
# vmax = heatmap_img.max()
# vmin = heatmap_img.min()
print(heatmap_img.max())
print(heatmap_img.min())
print(vmin, vmax)

if np.abs(vmax) > np.abs(vmin):
    vmax = np.abs(vmax)
    vmin = -np.abs(vmax)
else:
    vmax = np.abs(vmin)
    vmin = -np.abs(vmin)

thresh_max = vmax / 10 * 5
thresh_min = vmin / 10 * 5

# thresh_max = np.percentile(heatmap_img, 97)
# thresh_min = np.percentile(heatmap_img, 3)
# print(thresh_min, thresh_max)
# if np.abs(thresh_max) < np.abs(thresh_min):
#     thresh_max = np.abs(thresh_max)
#     thresh_min = -np.abs(thresh_max)
# else:
#     thresh_max = np.abs(thresh_min)
#     thresh_min = -np.abs(thresh_min)

alpha = 0.5
axes = []
for j, q in enumerate(axis_type):
    for i, p in enumerate(list_interval[j]):

        ax1 = fig.add_subplot(gs[j, i])

        if j == 0:
            orig_scattering_img = np.asarray(orig_img[int(p), :, :])
            heatmap_scattering_img = np.asarray(heatmap_img[int(p), :, :])
        elif j == 1:
            orig_scattering_img = np.asarray(orig_img[:, int(p), :])
            heatmap_scattering_img = np.asarray(heatmap_img[:, int(p), :])
        elif j == 2:
            orig_scattering_img = np.asarray(orig_img[:, :, int(p)])
            heatmap_scattering_img = np.asarray(heatmap_img[:, :, int(p)])

        orig_scattering_img = np.rot90(orig_scattering_img)
        heatmap_scattering_img = np.rot90(heatmap_scattering_img)
        heatmap_scattering_img[(heatmap_scattering_img < thresh_max) *(heatmap_scattering_img > thresh_min)] = np.nan

        if i == 0:
            # ax1.set_title(axis_type[j])
            ax1.set_ylabel(axis_type[j])
            # plt.ylabel(axis_type[j])
        ax1.imshow(orig_scattering_img, cmap=cmap_orig, vmin=orig_vmin, vmax=orig_vmax)
        # im = ax1.imshow(heatmap_scattering_img, cmap=cmap_heatmap, alpha=alpha, vmin=positive_vmin, vmax=positive_vmax)
        im = ax1.imshow(heatmap_scattering_img, cmap=cmap_heatmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        axes.append(ax1)
        # ax1.axis('off')
        del orig_scattering_img, heatmap_scattering_img

# (left, bottom, width, height)
cax = plt.axes([0.95, 0.1, 0.01, 0.8])
cbar = fig.colorbar(im, ax=axes, extend='both', cax=cax)

cbar.set_ticks(np.array((vmin, thresh_min, thresh_max, vmax)))
cbar.set_ticklabels(["%.2f" % (vmin), "%.2f" % (thresh_min), "%.2f" % (thresh_max), "%.2f" % (vmax)])
# plt.subplots_adjust(bottom=0.1, right=0.6, top=0.9, left=0.5)

plt.tight_layout()
plt.savefig('./a_test.png')
plt.close('all')
