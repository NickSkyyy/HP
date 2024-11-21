from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 24
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

fig, ax = plt.subplots()
ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower left")
ax_inset.xaxis.set_label_position("top")  # 将 x 轴标签设置在上方
ax_inset.xaxis.tick_top()  # 将x轴刻度移动到上方
ax_inset.yaxis.set_label_position("right")  # 将y轴标签放到右侧
ax_inset.yaxis.tick_right()  # 将y轴刻度移动到右侧
x1, x2 = 2, 10

CLICK = int(1e5)

x = np.linspace(2, 1e4, CLICK)

# Ours
# 3rd p (default)
p = lambda x: np.log2(x) / (x - 1)
m = lambda x: x * (x - 1) / 2 * p(x)
d = lambda x: 2 * m(x) / x

parse = lambda x: x / d(x) * np.log2(x)
connect = lambda x: x / d(x) * x
rest = lambda x: np.log2(m(x)) * x + m(x) * np.log2(x)

fours = lambda x: np.log(parse(x) + connect(x) + rest(x))
yours = fours(x)
ax.plot(x, yours, label=r"$P_E = \frac{\log N}{N - 1}$", color="#557AA4")
ax_inset.plot(x, yours, color="#557AA4")


# 4th p
p = lambda x: 2 * np.log2(x) / (x - 1)
m = lambda x: x * (x - 1) / 2 * p(x)
d = lambda x: 2 * m(x) / x

parse = lambda x: x / d(x) * np.log2(x)
connect = lambda x: x / d(x) * x
rest = lambda x: np.log2(m(x)) * x + m(x) * np.log2(x)

fours = lambda x: np.log(parse(x) + connect(x) + rest(x))
yours = fours(x)
ax.plot(x, yours, label=r"$P_E = \frac{2\log N}{N - 1}$", color="#86A0BE")
ax_inset.plot(x, yours, color="#86A0BE")


# define m
p = 0.2358
m = lambda x: x * (x - 1) / 2 * p
d = lambda x: 2 * m(x) / x

parse = lambda x: x / d(x) * np.log2(x)
connect = lambda x: x / d(x) * x
rest = lambda x: np.log2(m(x)) * x + m(x) * np.log2(x)

fours = lambda x: np.log(parse(x) + connect(x) + rest(x))
yours = fours(x)
ax.plot(x, yours, label=r"$M = cM_{\mathcal{G}}$", color="#B63D3D")
ax_inset.plot(x, yours, color="#B63D3D")
y2 = fours(x2)


# N**2 replace
p = 0.2358
m = lambda x: x * (x - 1) / 2 * p
d = lambda x: 2 * m(x) / x

parse = lambda x: x / d(x) * np.log2(x)
connect = lambda x: x / d(x) * x
rest = lambda x: x**2

fours = lambda x: np.log(parse(x) + connect(x) + rest(x))
yours = fours(x)
ax.plot(x, yours, label=r"$N^2$ replace", color="#F7DEDB")
ax_inset.plot(x, yours, color="#F7DEDB")
y2 = fours(x2)


# N**2
fea = lambda x: np.log(x**2)
yea = fea(x)
ax.plot(x, yea, label=r"Plain: $\mathcal{O}(N^2)$", color="#000000")
ax_inset.plot(x, yea, color="#000000")
y1 = fea(x1)

# ===================================================================================
# best M
# import sympy as sp
# from scipy.optimize import fsolve
# def f(x, m):
#   d = 2 * m / x

#   parse = x / d * np.log2(x)
#   connect = (x / d)**2
#   rest = x * np.log2(m) + m * np.log2(x)

#   return np.log2(parse + connect + rest)

# def g(x):
#   return np.log2(x**2)

# def solve_for_m(x):
#   initial_guess = 1.0
#   m_solution = fsolve(lambda m: f(x, m) - g(x), initial_guess)
#   return m_solution

# x = 100
# m = solve_for_m(x)
# print(m)
# input()
# ===================================================================================


ax.legend(loc="lower right")
ax.set_xlim(left=2)

# 在小图中绘制局部数据
ax_inset.set_xlim(x1, x2)
ax_inset.set_ylim(y1, y2)

# 小图添加标注
ax_inset.set_xlim(x1, x2)
ax_inset.set_ylim(y1, y2)
ax.grid(which="major", linestyle="--", linewidth=0.5)


ax.set_xlabel(r"# Nodes ($N$)")
ax.set_ylabel(r"Log Time ($\log y$)")
plt.subplots_adjust(top=0.5, left=0.5, right=0.75)

plt.grid(True)
plt.show()