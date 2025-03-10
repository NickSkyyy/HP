from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

plt.rcParams["font.size"] = 24
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

categories = ['Flickr', 'PPI']
data1 = [2.10e-2, 7.60e-1, 8.90e-1]
data1 = [x / 1.70e0 for x in data1]
data2 = [5.70e-4, 5.85e-3, 1.02e-1]
data2 = [x / 1.09e-1 for x in data2]

stacked_data1 = np.array(data1)
stacked_data2 = np.array(data2)

plt.barh(categories[0], stacked_data1[0], color='#557AA4', label='T1')
plt.barh(categories[0], stacked_data1[1], left=stacked_data1[0], color='#86A0BE', label='T2')
plt.barh(categories[0], stacked_data1[2], left=stacked_data1[0] + stacked_data1[1], color='#CED7E4', label='T3')

plt.barh(categories[1], stacked_data2[0], color='#B63D3D', label='T1')
plt.barh(categories[1], stacked_data2[1], left=stacked_data2[0], color='#E39B96', label='T2')
plt.barh(categories[1], stacked_data2[2], left=stacked_data2[0] + stacked_data2[1], color='#F7DEDB', label='T3')

plt.title('Detailed Runtime Analysis')
plt.xlabel('Times (%)')

plt.legend()

plt.show()