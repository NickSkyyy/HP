from eval.stats import *
from math import *
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import *

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

import networkx as nx
import numpy as np
import random as r

methods = [
  "EA",
  "BA",
  "WR",
  "MMSB",
  "Kronecker",
  "Ours",
]

def draw(graphs, dataset, name):
  # if not os.path.exists(".\\visual\\%s" % dataset):
  #   os.mkdir(".\\visual\\%s" % dataset)
  # for i, G in enumerate(graphs):
  #   colors = ["#557AA4" for _ in range(G.number_of_nodes())]
  #   nx.draw(G, pos=nx.spring_layout(G), node_color=colors, node_size=100)
  #   plt.savefig(".\\visual\\%s\\%d_%s.png" % (dataset, i, name))
  #   # plt.show()
  #   plt.close()
  pass

def eval(Gref, Gpred):
  # print("%.4lf" % degree_stats(Gref, Gpred))
  # print("%.4lf" % clustering_stats(Gref, Gpred))
  # print("%.4lf" % orbit_stats_all(Gref, Gpred))
  pass

def degrees(graphs):
  deg = []
  cnt = 0
  tlen = 0
  for graph in graphs:
    try:
      bar = nx.degree_histogram(graph)
      tlen += 1    
      if len(bar) > cnt:
        cnt = len(bar)
      deg.append(bar)
    except:
      continue
  for i in range(tlen):
    deg[i] = deg[i] + [0 for _ in range(cnt - len(deg[i]))]
  deg = np.array(deg)
  print(deg.shape)
  # input()
  deg = np.mean(deg, axis=0)
  print(deg.shape)
  return deg

def load_data(dataset, root=".\data"):
  print("load dataset %s" % dataset)
  graphs = None
  if dataset in ["MUTAG", "NCI1", "ENZYMES", "DD", "PROTEINS",
                 "IMDB-BINARY", "REDDIT-BINARY", "IMDB-MULTI", "deezer_ego_nets", ]:
    graphs = TUDataset(root, dataset)
    graphs = [to_networkx(graph, to_undirected=True) for graph in graphs]
  elif dataset == "GRID":
    graphs = []
    for _ in range(200):
      x = r.randint(10, 20)
      y = r.randint(10, 20)   
      graphs.append(nx.grid_2d_graph(x, y))
  elif dataset == "TREE":
    graphs = []
    for _ in range(200):
      n = r.randint(100, 200)
      graphs.append(nx.random_tree(n))
  elif dataset == "RAND":
    graphs = []
    for _ in range(200):
      n = r.randint(100, 200)
      m = r.randint(n - 1, n * (n - 1) / 2)
      graphs.append(nx.dense_gnm_random_graph(n, m))
  elif dataset == "CLUS":
    graphs = []
    for _ in range(200):
      n = r.randint(100, 200)
      m = r.randint(n - 1, n * (n - 1) / 2)
      p = 2 * m / (n * (n - 1))
      m = max(1, int(2 * m / n))
      graphs.append(nx.powerlaw_cluster_graph(n, m, p))
  elif dataset == "EGO":
    graphs = []
    while len(graphs) < 200:
      try:
        n = 150
        graphs.append(nx.random_powerlaw_tree(n))
      except:
        continue
  elif dataset == "TEST":
    graphs = []
    n = 150
    p = log2(n) / (n - 1)
    # p = 0.1
    m = int(n * (n - 1) / 2 * p)
    graphs.append(nx.dense_gnm_random_graph(n, m))
  return graphs

def poisson(k, mu, maxlim=-1):
  """
  ### Params
  - k: possibility of X = k
  - mu: the lambda
  """
  # ==========================================================================
  # cutoffs when too big
  # lim = 1
  # p0 = pow(mu, 0) * exp(-mu) / factorial(0)
  # while True:
  #   val = pow(mu, lim) * exp(-mu) / factorial(lim)
  #   if val <= p0:
  #     break
  #   lim += 1
  # if k >= lim:
  #   return 0.0
  if maxlim != -1 and k >= maxlim:
    return 0.0
  # ==========================================================================
  return pow(mu, k) * exp(-mu) / factorial(k)

# def poisson_sampler(n, m, size=1):
#   """
#   Sample `size` node in poisson distribution within `n`
#   """
#   if n == 1:
#     if size == 1:
#       return 0
#     else:
#       return [0 for _ in range(size)]
#   mu = 2 * m / n
#   rr = min(n - 1, m) + 1
#   plist = [poisson(i, mu) for i in range(1, rr)]
#   plist = [val / sum(plist) for val in plist] 
#   samples = np.random.choice(np.arange(1, rr), size=int(size), p=plist)
#   if size == 1:
#     return samples[0]
#   else:
#     return samples

def poisson_sampler(n, mu, size=1, maxlim=-1):
  """
  sample degree from n nodes
  ### Params
  - n: node that `d=n-1`
  """
  plist = [poisson(i, mu, maxlim) for i in range(n)]
  plist = [val / sum(plist) for val in plist] 
  samples = np.random.choice(np.arange(n), size=int(size), p=plist)
  if size == 1:
    return samples[0]
  else:
    return samples
  
def poisson_fast_sampler(plist, size=1):
  """
  sample degree from n nodes
  ### Params
  - n: node that `d=n-1`
  """
  samples = np.random.choice(np.arange(len(plist)), size=int(size), p=plist)
  if size == 1:
    return samples[0]
  else:
    return samples
  

# different distribution sampler
# ===================================================================================
def random_sampler(n, p, size=1, replace=True):
  if n == 0:
    return 0
  samples = np.random.choice(np.arange(n), size=int(size), p=p, replace=replace)
  if size == 1:
    return samples[0]
  else:
    return samples
  
def uni_sampler(n, size=1, replace=False):
  if n == 0:
    return 0
  samples = np.random.choice(np.arange(n), size=size, replace=replace)
  if size == 1:
    return samples[0]
  else:
    return samples
  
import scipy.stats as stats
def norm(k):
  return stats.norm.cdf(k)

def norm_fast_sampler(plist, size=1):
  samples = np.random.choice(np.arange(len(plist)), size=int(size), p=plist)
  if size == 1:
    return samples[0]
  else:
    return samples

def expp(k, D):
  return stats.expon.cdf(k, scale=D)

def gamma(k, a, b):
  return stats.gamma.cdf(k, a, scale=b)

def pareto(k, D, p=1):
  return stats.pareto.cdf(k, D, scale=p)
# ===================================================================================

# Kronecker
def kronecker(N, D):
  G0 = [[0 for i in range(D + 1)] for j in range(D + 1)]
  for j in range(D + 1):
    G0[0][j] = 1
    G0[j][0] = 1
    G0[j][j] = 1
  cnt = D + 1
  while cnt < N:
    tN = len(G0)
    tG = [[0 for i in range(tN**2)] for j in range(tN**2)]
    for i in range(tN):
      for j in range(tN):
        if G0[i][j] == 1:
          tx = tN * i
          ty = tN * j
          for ti in range(tx, tx + tN):
            for tj in range(ty, ty + tN):
              tG[ti][tj] = G0[ti - tx][tj - ty]
    G0 = tG
    cnt = len(tG)
  for i in range(len(G0)):
    G0[i][i] = 0
  G0 = np.array(G0)[:N, :N]
  G = nx.from_numpy_array(G0)
  return G


# MMSB
class MMSB:  
  def __init__(self, num_nodes, num_communities, p_init=0.5):  
    self.num_nodes = num_nodes  
    self.num_communities = num_communities  
    self.p_init = p_init 
      
    self.membership_probs = np.full((num_nodes, num_communities), p_init / num_communities)  
        
    self.community_connection_probs = np.random.rand(num_communities, num_communities)  
    self.normalize_community_connection_probs()  
        
  def normalize_community_connection_probs(self):    
    row_sums = np.sum(self.community_connection_probs, axis=1, keepdims=True)  
    self.community_connection_probs /= row_sums  
        
  def generate_graph(self):  
    G = nx.Graph()  
    for i in range(self.num_nodes):  
      for j in range(i + 1, self.num_nodes):  
        edge_prob = np.dot(self.membership_probs[i], np.dot(self.community_connection_probs, self.membership_probs[j]))          
        if np.random.rand() < edge_prob:  
          G.add_edge(i, j)   
    return G