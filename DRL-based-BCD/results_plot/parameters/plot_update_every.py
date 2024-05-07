import pandas as pd
import seaborn as sns
import seaborn.timeseries
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

start = 1
end = 0
my_list = list()
df=pd.read_csv('update_every.csv', header=None, sep=',')
for i in range(start,len(df)-end):
    my_list.append(list(df.iloc[i]))

my_arr = np.array(my_list).T
arr1 = my_arr[0]
arr2 = my_arr[1]
arr3 = my_arr[2]
arr4 = my_arr[3]
arr5 = my_arr[4]
# mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.
mode = "interp"
# arr1 = savgol_filter(arr1, 5, 3, mode= mode)
# arr2 = savgol_filter(arr2, 5, 3, mode= mode)
# arr3 = savgol_filter(arr3, 5, 3, mode= mode)
# arr4 = savgol_filter(arr4, 5, 3, mode= mode)
# arr5 = savgol_filter(arr5, 5, 3, mode= mode)
window_zise = 2
arr3 = moving_average(interval = arr3, window_size = window_zise)
arr2 = moving_average(interval = arr2, window_size = 3)
arr4 = moving_average(interval = arr4, window_size = 3)

arr1 = moving_average(interval = arr1, window_size = window_zise)
arr5 = moving_average(interval = arr5, window_size = 2)
plt.figure(figsize=(7.6, 5.7))
plt.plot(arr1,color="green", linewidth = 1.5)
plt.plot(arr2,color="goldenrod",linewidth = 1.5)
plt.plot(arr3,color="b",linewidth = 1.5)
plt.plot(arr4,color="gray",linewidth = 1.5)
plt.plot(arr5,color="tomato",linewidth = 1.5)
plt.plot(arr3,color="b",linewidth = 1.5)
fontsize = 16
axis_size = 10
label = ["update_every = 50","update_every = 100","update_every = 200","update_every = 500","update_every = 800"]
plt.legend(labels=label, fontsize=fontsize)
plt.xlabel("Episode", fontsize=fontsize)
plt.ylabel("Reward", fontsize=fontsize)
plt.xticks(fontsize=11)
plt.yticks(fontsize=axis_size)
plt.savefig("update_every.pdf")
plt.show()