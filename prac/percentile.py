import numpy as np

a = np.random.randint((50), size=(10))
# a = np.array((1, 3, 5))
print(np.sort(a))
a_min = a.min()
a_max = a.max()
print((a_max - a_min) * 0.9 + a_min)


percentile = 90
thresh_hold = np.percentile(np.sort(a, axis=None), percentile)
print(thresh_hold)
thresh_hold = np.percentile(a, percentile)
print(thresh_hold)