from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# 24-55
plt.rcParams["font.size"] = 48
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

plt.barh(categories[0], stacked_data1[0], color='#C66218', label='parsing')
plt.barh(categories[0], stacked_data1[1], left=stacked_data1[0], color='#FFBF00', label='connecting')
plt.barh(categories[0], stacked_data1[2], left=stacked_data1[0] + stacked_data1[1], color='#004285', label='rebuilding')

plt.barh(categories[1], stacked_data2[0], color='#C66218')
plt.barh(categories[1], stacked_data2[1], left=stacked_data2[0], color='#FFBF00')
plt.barh(categories[1], stacked_data2[2], left=stacked_data2[0] + stacked_data2[1], color='#004285')

plt.title('Detailed Runtime Analysis')
plt.xlabel('Times (%)')

plt.legend()

plt.show()