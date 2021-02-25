import numpy as np

"""
[3, 5, 9, 17]
[1, 2, 4, 12]
featureMap size : 10
featureMap dim : 1000
finished

[3, 5, 9, 17, 33]
[1, 2, 4, 8, 16]
featureMap size : 6
featureMap dim : 216
finished

[7, 11, 19, 35, 67]
[2, 4, 8, 16, 16]
featureMap size : 4
featureMap dim : 64
finished
"""

""" code for calculating the receptive field size of featuremaps"""
input_size = 170
# list_conv_kernel = [3, 3, 3, 3, 3, ] # kernel size of the layers
# list_conv_stride = [2, 1, 2, 2, 1, ] # stride size of the layers
list_conv_kernel = [3, 3, 3, 3, 3, 3] # kernel size of the layers
list_conv_stride = [2, 2, 1, 1, 1, 1] # stride size of the layers
# list_conv_kernel = [7, 3, 3, 3] # kernel size of the layers
# list_conv_stride = [2, 2, 2, 1] # stride size of the layers
list_rf = []
list_rf_diff = []

for i in range(len(list_conv_kernel)):
    if len(list_rf) == 0:
        list_rf.append(list_conv_kernel[i])
        list_rf_diff.append(list_conv_stride[i])
    else:
        current_rf = list_rf[i-1] + (list_rf_diff[i-1] * (list_conv_kernel[i] - 1))
        current_rf_diff = list_rf_diff[i-1] * list_conv_stride[i]

        list_rf.append(current_rf)
        list_rf_diff.append(current_rf_diff)

featureMap_size = (input_size - list_rf[-1]) // list_rf_diff[-1] + 1

print(list_rf)
print(list_rf_diff)
print("featureMap size : {}".format(featureMap_size))
print("featureMap dim : {}".format(pow(featureMap_size, 3)))
print("finished")
print()
