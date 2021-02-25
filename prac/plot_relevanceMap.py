import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import os
import sys
list_file_name = [
# 'relevance_NC_logit_0.nii.gz',
# 'relevance_NC_logit_1.nii.gz',
#
# 'relevance_sMCI_logit_0.nii.gz',
# 'relevance_sMCI_logit_1.nii.gz',
#
# 'relevance_pMCI_logit_0.nii.gz',
# 'relevance_pMCI_logit_1.nii.gz',
#
# 'relevance_AD_logit_0.nii.gz',
# 'relevance_AD_logit_1.nii.gz',

'LRLC.nii.gz',
'featuremap.nii.gz',
]

for tmp_i in range(len(list_file_name)):
    file_dir_orig = '../featuremap/' + list_file_name[tmp_i]
    save_file_name = list_file_name[tmp_i][:-7]

    orig_img = nib.load(file_dir_orig).get_data()
    shape = orig_img.shape

    list_interval = []
    for j in range(3):
        tmp_list = []
        for i in np.arange(20, 81, 4):
            tmp_list.append(int(np.percentile(np.arange(0, shape[j]), i)))
        list_interval.append(np.hstack(tmp_list))

    axis_type = ['Sagittal', 'Coronal', 'Axial']

    fig = plt.figure(figsize=(list_interval[0].shape[0] * 3, len(axis_type) * 3))
    plt.rcParams.update({'font.size': 30})
    fig.suptitle(save_file_name, fontsize=30)

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
    if tmp_i == 0 :
        orig_vmax = np.percentile(orig_img, 99)
        orig_vmin = np.percentile(orig_img, 1)
        print(orig_vmin, orig_vmax)

        if np.abs(orig_vmax) > np.abs(orig_vmin):
            orig_vmax = np.abs(orig_vmax)
            orig_vmin = -np.abs(orig_vmax)
        else:
            orig_vmax = np.abs(orig_vmin)
            orig_vmin = -np.abs(orig_vmin)

        thresh_max = orig_vmax / 10 * 5
        thresh_min = orig_vmin / 10 * 5

    alpha = 0.5
    axes = []
    for j, q in enumerate(axis_type):
        for i, p in enumerate(list_interval[j]):

            ax1 = fig.add_subplot(gs[j, i])

            if j == 0:
                orig_scattering_img = np.asarray(orig_img[int(p), :, :])
            elif j == 1:
                orig_scattering_img = np.asarray(orig_img[:, int(p), :])
            elif j == 2:
                orig_scattering_img = np.asarray(orig_img[:, :, int(p)])

            orig_scattering_img = np.rot90(orig_scattering_img)

            if i == 0:
                # ax1.set_title(axis_type[j])
                ax1.set_ylabel(axis_type[j])
                # plt.ylabel(axis_type[j])
            im = ax1.imshow(orig_scattering_img, cmap=cmap_orig, vmin=orig_vmin, vmax=orig_vmax)
            # im = ax1.imshow(heatmap_scattering_img, cmap=cmap_heatmap, alpha=alpha, vmin=positive_vmin, vmax=positive_vmax)
            ax1.set_yticks([])
            ax1.set_xticks([])
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            axes.append(ax1)
            del orig_scattering_img
            # ax1.axis('off')

    # (left, bottom, width, height)
    cax = plt.axes([0.95, 0.1, 0.01, 0.8])
    cbar = fig.colorbar(im, ax=axes, extend='both', cax=cax)

    cbar.set_ticks(np.array((orig_vmin, thresh_min, thresh_max, orig_vmax)))
    cbar.set_ticklabels(["%.2f" % (orig_vmin), "%.2f" % (thresh_min), "%.2f" % (thresh_max), "%.2f" % (orig_vmax)])
    # plt.subplots_adjust(bottom=0.1, right=0.6, top=0.9, left=0.5)

    plt.tight_layout()
    plt.savefig('./' + save_file_name)
    plt.close('all')
