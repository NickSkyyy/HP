from torch_geometric.data import Data
from torch_geometric.utils import *
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300

import itertools
import networkx as nx
import time

from args import args
from util import *

import numpy as np
import random as r
# candidate
# 78, 
SEED = 78
np.random.seed(SEED)
r.seed(SEED)

NUM = 1
UNDER_LIM = True

if __name__ == "__main__":
  os.makedirs("visual/{}".format(args.dataset), exist_ok=True)
  graphs = load_data(args.dataset)[:1]
  deg = 0
  print(len(graphs))
  # draw(graphs, args.dataset, "ori")
  # exit()

  if "Ours" in methods:
    print("Ours")
    GOurs = []
    times = 0
    for graph in tqdm(graphs):
      # 1. prepare the statistics
      N, M = len(graph.nodes()), len(graph.edges())
      D = 2 * M / N      
      # ===========================================================
      maxlim = -1
      lim = 1
      p0 = pow(D, 0) * exp(-D) / factorial(0)
      while True:
        val = pow(D, lim) * exp(-D) / factorial(lim)
        if val <= p0:
          break
        lim += 1
      # cut off when under 0
      if UNDER_LIM:
        maxlim = lim
      # maxlim = MAXD
      # ===========================================================
      G = graph
      # print(nx.degree_histogram(G))
      MAXD = len(nx.degree_histogram(G)) - 1
      print(MAXD)
      if 2 * M == N * (N - 1):
        G = nx.complete_graph(N)
        GOurs.append(G)
        continue
      
      # 2. pick (d+1, d), d = MAXD
      if UNDER_LIM:
        subs = [(MAXD + 1, MAXD)]
        sN, sM = N - MAXD - 1, M - MAXD - 1
        sumN, sumM = MAXD + 1, MAXD
      else:
        subs = []
        sN, sM = N, M
        sumN, sumM = 0, 0
      poi_plist = [poisson(k, D, maxlim) for k in range(0, N)]
      poi_plist = [val / sum(poi_plist) for val in poi_plist]
      start_time = time.time()
      asampler = Alias(poi_plist)
      while sN > 0 and sM > 0:
        # nn = min(sN, MAXD + 1)
        # d = poisson_sampler(nn, D)
        # d = poisson_fast_sampler(poi_plist, 1)
        d = asampler.sample()
        if d > sN:
          d = sN
        if UNDER_LIM and d > MAXD:
          d = MAXD
        subs.append((d + 1, d))
        sN -= d + 1
        sM -= d + 1
        sumN += d + 1
        sumM += d
      if sM <= 0 and sN > 0:
        subs.extend([(1, 0) for _ in range(sN)])
        sumN += sN
      end_time = time.time()
      times += end_time - start_time

      # 3. build edges between nodes
      # 3.0 prepare
      # start_time = time.time()
      entries = [[sub[1]] + [1 for _ in range(sub[1])] for sub in subs]

      # 3.1 make sure to connect
      edges = []
      if len(subs) > 1:
        for i, sub in enumerate(subs):
          if i == 0:
            continue

          if UNDER_LIM:
            goi = [poisson(x + 1, D, maxlim) if x < MAXD else 0 for x in entries[i]]
          else:
            goi = [poisson(x + 1, D, maxlim) for x in entries[i]]
          if sum(goi) == 0:
            continue
          if UNDER_LIM:
            goi[0] = 0 if sum(goi[1:]) != 0 else goi[0]
          goi = [val / sum(goi) for val in goi]
          ll = random_sampler(len(goi), goi)

          j = r.randint(0, i - 1)
          if UNDER_LIM:
            goj = [poisson(x + 1, D, maxlim) if x < MAXD else 0 for x in entries[j]]
          else:
            goj = [poisson(x + 1, D, maxlim) for x in entries[j]]
          if sum(goj) == 0:
            continue
          if UNDER_LIM:
            goj[0] = 0 if sum(goj[1:]) != 0 else goj[0]
          goj = [val / sum(goj) for val in goj]
          rr = random_sampler(len(goj), goj)

          entries[i][ll] += 1
          entries[j][rr] += 1
          edges.append(((i, ll), (j, rr)))
          if len(edges) == sM:
            break
      end_time = time.time()
      times += end_time - start_time

      # ===========================================================================
      # build graph
      node_id = 0
      indice = []
      edge_index = []
      for sub in subs:
        indice.append(node_id)
        for _ in range(sub[1]):
          node_id += 1
          edge_index.append((indice[-1], node_id))
        node_id += 1
      G = nx.empty_graph()
      G.add_nodes_from(indice)
      G.add_edges_from(edge_index)

      colors = ["#C66218" if node in indice else "#004285" for node in G.nodes()]
      print(colors)
      nx.draw(G, pos=nx.spring_layout(G), node_color=colors, node_size=100, with_labels=True)
      plt.savefig(".\\visual\\%s\\parsing stage.png" % (args.dataset))
      plt.close()

      edge_index = []
      for edge in edges:
        u = indice[edge[0][0]] + edge[0][1]
        v = indice[edge[1][0]] + edge[1][1]
        edge_index.extend([(u, v), (v, u)])
      G.add_edges_from(edge_index)
      edge_color = ["#C66218" if edge in edge_index else "black" for edge in G.edges()]
      pN, pM = len(G.nodes()), len(G.edges())

      print(edge_index)
      colors = ["#C66218" if node in indice else "#004285" for node in G.nodes()]
      nx.draw(G, pos=nx.spring_layout(G), node_color=colors, node_size=100, with_labels=True,  edgelist=G.edges(), edge_color=edge_color)
      plt.savefig(".\\visual\\%s\\connecting stage.png" % (args.dataset))
      plt.close()

      # 3.2 basic entries
      res_edges = M - pM
      extra_edges = []
      cnt = res_edges
      while cnt > 0 and res_edges > 0:
        # print(entries)
        # ====================================================================
        # inner update
        nlist = []
        plist = []
        for sub, elist in enumerate(entries):
          psub = len(elist) / sumN
          sublist = []
          for eid, edeg in enumerate(elist):
            # pick under MAXD
            if UNDER_LIM:
              p = psub * (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
            # pick without limit
            else:
              p = psub * (poisson(edeg + 1, D, maxlim))
            nlist.append((sub, eid))
            sublist.append(p)
          # pick without limit
          if UNDER_LIM:
            sublist[0] = 0 if sum(sublist[1:]) != 0 else sublist[0]
          plist.extend(sublist)
        # ====================================================================
        sumt = sum(plist)
        plist = [p / sumt for p in plist]
        asampler = Alias(plist)

        # V2 of picking
        eset = set()
        for _ in range(res_edges):
          # u, v = random_sampler(len(nlist), p=plist, size=2, replace=False)
          # u, v = asampler.sample(2)
          while True:
            u = asampler.sample()
            v = asampler.sample()
            if u != v:
              break
          u, v = min(u, v), max(u, v)
          eset.add((u, v))
        res_edges -= len(eset)
        cnt = min(cnt // 2, res_edges)

        while len(eset) != 0:
          u, v = eset.pop()
          u, uentry = nlist[u][0], nlist[u][1]
          v, ventry = nlist[v][0], nlist[v][1]
          entries[u][uentry] += 1
          entries[v][ventry] += 1
          extra_edges.append((indice[u] + uentry, indice[v] + ventry))
        # ======================================================================
      
      G.add_edges_from(extra_edges)
      edge_color = ["#C66218" if edge in extra_edges else "black" for edge in G.edges()]
      nx.draw(G, pos=nx.spring_layout(G), node_color=colors, node_size=100, with_labels=True,  edgelist=G.edges(), edge_color=edge_color)
      plt.savefig(".\\visual\\%s\\rebuilding stage.png" % (args.dataset))
      # plt.show()
      plt.close()
      GOurs.append(G)
      if len(GOurs) == NUM:
        break
    del GOurs

  del graphs