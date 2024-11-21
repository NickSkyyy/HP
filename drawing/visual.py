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

fig = plt.figure()
hrate = 0
gs = GridSpec(6, 9, figure=fig, hspace=-hrate, wspace=0.3)
# gs = GridSpec(2, 3, figure=fig, wspace=0.3)

data = [
  "ENZYMES",
  "MUTAG",
  "NCI",
  "PROTEINS",
]
img_dir = ".\\visual\\"
lx = 0.80
ly = 0.85
px = 0.55
py = 0.05

for i in range(len(data)):
  name = data[i]
  x, y = 1, 0
  x += i

  Gori = mpimg.imread(os.path.abspath(img_dir + "{}_ori.png".format(name)))
  Gsota = mpimg.imread(os.path.abspath(img_dir + "{}_sota.png".format(name)))
  Gour = mpimg.imread(os.path.abspath(img_dir + "{}_ours.png".format(name)))

  ax = fig.add_subplot(gs[x, y])
  ax.imshow(Gori, aspect="auto")
  ax.axis("off")

  ax = fig.add_subplot(gs[x, y + 1])
  ax.imshow(Gour, aspect="auto")
  ax.axis("off")
  
  ax = fig.add_subplot(gs[x, y + 2])
  ax.imshow(Gsota, aspect="auto")
  ax.axis("off")
ax = fig.add_subplot(gs[5, 0:2])
ax.text(lx, ly, "Bioinformatics & Molecules", ha="center")
ax.axis("off")

data = [
  "deezer",
  "IB",
  "IM",
  "RB"
]
for i in range(len(data)):
  name = data[i]
  x, y = 1, 3
  x += i

  Gori = mpimg.imread(os.path.abspath(img_dir + "{}_ori.png".format(name)))
  Gsota = mpimg.imread(os.path.abspath(img_dir + "{}_sota.png".format(name)))
  Gour = mpimg.imread(os.path.abspath(img_dir + "{}_ours.png".format(name)))

  ax = fig.add_subplot(gs[x, y])
  ax.imshow(Gori, aspect="auto")
  ax.axis("off")

  ax = fig.add_subplot(gs[x, y + 1])
  ax.imshow(Gour, aspect="auto")
  ax.axis("off")
  
  ax = fig.add_subplot(gs[x, y + 2])
  ax.imshow(Gsota, aspect="auto")
  ax.axis("off")
ax = fig.add_subplot(gs[5, 3:5])
ax.text(lx, ly, "Social Networks", ha="center")
ax.axis("off")

data = [
  "EGO",
  "TREE",
  "CLUS",
  "GRID",
]
for i in range(len(data)):
  name = data[i]
  x, y = 1, 6
  x += i

  Gori = mpimg.imread(os.path.abspath(img_dir + "{}_ori.png".format(name)))
  Gsota = mpimg.imread(os.path.abspath(img_dir + "{}_sota.png".format(name)))
  Gour = mpimg.imread(os.path.abspath(img_dir + "{}_ours.png".format(name)))

  ax = fig.add_subplot(gs[x, y])
  ax.imshow(Gori, aspect="auto")
  ax.axis("off")

  ax = fig.add_subplot(gs[x, y + 1])
  ax.imshow(Gour, aspect="auto")
  ax.axis("off")
  
  ax = fig.add_subplot(gs[x, y + 2])
  ax.imshow(Gsota, aspect="auto")
  ax.axis("off")
ax = fig.add_subplot(gs[5, 6:8])
ax.text(lx, ly, "Synthetic", ha="center")
ax.axis("off")

# set text
for i in range(0, 9, 3):
  ax = fig.add_subplot(gs[0, i])
  ax.text(px, py, "Real Graph", ha="center")
  ax.axis("off")

  ax = fig.add_subplot(gs[0, i+1])
  ax.text(px, py, "Ours", ha="center")
  ax.axis("off")

  ax = fig.add_subplot(gs[0, i+2])
  ax.text(px, py, "Baselines", ha="center")
  ax.axis("off")

# plt.subplots_adjust(top=1 - hrate)
plt.show()