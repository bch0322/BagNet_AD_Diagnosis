""" flags """
""" preprocessing for the data """
flag_orig_npy = False
flag_fold_index = False
flag_orig_npy_fold = False
flag_orig_npy_other_dataset = False
flag_print_trainAcc = False

""" flag whether to use pre-train model"""
flag_classification = True
flag_classification_using_pretrained = False
flag_classification_fine_tune = False
flag_regression = False

""" loss """
flag_loss_1 = False
flag_loss_2 = False
flag_loss_3 = True
flag_loss_4 = False

""" training strategy"""
flag_estimate_age = False
flag_ttest_save = False
flag_ttest_load = False
flag_downSample = False
flag_gaussian_norm_init = False
flag_gaussian_norm = False

"""
0 : always      
1 : no cropping       
2 : (1:1) cropping
"""

flag_translation = True
flag_eval_translation = False
flag_translation_ratio = 0

flag_cropping = False
flag_eval_cropping = False
flag_crop_ratio = 0

flag_Gaussian_blur = False



