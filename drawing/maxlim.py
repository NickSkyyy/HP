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
hrate = 0.05
gs = GridSpec(3, 5, figure=fig, hspace=hrate, wspace=0.03)
# gs = GridSpec(2, 3, figure=fig, wspace=0.3)

img_dir = ".\\"
lx = 0.90
ly = 0.50
px = 0.02
py = 0.02

# maxlim
Gp = mpimg.imread(os.path.abspath(img_dir + "lim_parse.png"))
Gl = mpimg.imread(os.path.abspath(img_dir + "lim_link.png"))
Gr = mpimg.imread(os.path.abspath(img_dir + "lim_rest.png"))

ax = fig.add_subplot(gs[1, 1])
ax.imshow(Gp, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 2])
ax.imshow(Gl, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 3])
ax.imshow(Gr, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

# no lim
Gp = mpimg.imread(os.path.abspath(img_dir + "no_parse.png"))
Gl = mpimg.imread(os.path.abspath(img_dir + "no_link.png"))
Gr = mpimg.imread(os.path.abspath(img_dir + "no_rest.png"))

ax = fig.add_subplot(gs[2, 1])
ax.imshow(Gp, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

ax = fig.add_subplot(gs[2, 2])
ax.imshow(Gl, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

ax = fig.add_subplot(gs[2, 3])
ax.imshow(Gr, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

# original
Gori = mpimg.imread(os.path.abspath(img_dir + "ori.png"))
ax = fig.add_subplot(gs[1, 4])
ax.imshow(Gori, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

ax = fig.add_subplot(gs[2, 4])
ax.imshow(Gori, aspect="auto")
ax.set_xticks([])  
ax.set_yticks([])

# set text
ax = fig.add_subplot(gs[1, 0])
ax.text(lx, ly, "under limitation", rotation=90, verticalalignment="center")
ax.axis("off")

ax = fig.add_subplot(gs[2, 0])
ax.text(lx, ly, "no limitation", rotation=90, verticalalignment="center")
ax.axis("off")

ax = fig.add_subplot(gs[0, 1])
ax.text(px, py, "1. parse for anchor", ha="left")
ax.axis("off")

ax = fig.add_subplot(gs[0, 2])
ax.text(px, py, "2. first connection", ha="left")
ax.axis("off")

ax = fig.add_subplot(gs[0, 3])
ax.text(px, py, "3. rest edges", ha="left")
ax.axis("off")

ax = fig.add_subplot(gs[0, 4])
ax.text(px, py, "real graph", ha="left")
ax.axis("off")

# plt.subplots_adjust(top=1 - hrate)
plt.show()