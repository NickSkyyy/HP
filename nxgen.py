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

NUM = 1000000

if __name__ == "__main__":
  graphs = load_data(args.dataset)[:args.num]
  if args.method != "all":
    methods = [args.method]
  # deg = 0
  # clus = 0
  # orbits = 0
  # ocnt = 0
  # sparse = 0
  # print(len(graphs))
  # draw(graphs, args.dataset, "ori")
  # for graph in graphs:
  #   nn = 0
  #   for val in nx.degree(graph):
  #     nn += val[1]
  #   deg += nn / len(graph.nodes())
  #   clus += nx.average_clustering(graph)
  #   sparse += (2 * len(graph.edges())) / (len(graph.nodes()) * (len(graph.nodes()) - 1))
  #   try:
  #     nn = orca(graph)
  #     orbits += np.mean(np.sum(nn, axis=0) / len(graph.nodes()))
  #     ocnt += 1
  #   except:
  #     print("oops")
  # deg = deg / len(graphs)
  # clus = clus / len(graphs)
  # sparse = sparse / len(graphs)
  # orbits = orbits / ocnt
  # print(deg)
  # print(clus)
  # print(orbits)
  # print(sparse)
  # input()

  if "ER" in methods:
    print("ER")
    GER = []
    times = 0
    for graph in tqdm(graphs):
      start_time = time.time()
      N, M = len(graph.nodes()), len(graph.edges())
      # p = 2 * log(N) / (N - 1)
      p = 2 * M / (N * (N - 1))
      G = nx.gnp_random_graph(N, p)
      end_time = time.time()
      times += end_time - start_time
      GER.append(G)
      if len(GER) == NUM:
        break
    draw(GER, args.dataset, "ER")
    eval(graphs, GER)
    print("total times: %f" % (times / len(graphs)))
    del GER

  if "BA" in methods:
    print("BA")
    GBA = []
    times = 0
    for graph in tqdm(graphs):
      start_time = time.time()
      N, M = len(graph.nodes()), len(graph.edges())
      # p = 2 * log(N) / (N - 1)
      D = max(1, int(2 * M / N))
      G = nx.barabasi_albert_graph(N, D)
      end_time = time.time()
      times += end_time - start_time
      GBA.append(G)
      if len(GBA) == NUM:
        break
    draw(GBA, args.dataset, "BA")
    eval(graphs, GBA)
    print("total times: %f" % (times / len(graphs)))
    del GBA

  if "WS" in methods:
    print("WS")
    GWS = []
    times = 0
    for graph in tqdm(graphs):
      start_time = time.time()
      N, M = len(graph.nodes()), len(graph.edges())
      D = max(1, int(2 * M / N))
      p = 2 * M / (N * (N - 1))
      G = nx.watts_strogatz_graph(N, D, p)
      end_time = time.time()
      times += end_time - start_time
      GWS.append(G)
      if len(GWS) == NUM:
        break
    draw(GWS, args.dataset, "WS")
    eval(graphs, GWS)
    print("total times: %f" % (times / len(graphs)))
    del GWS

  if "MMSB" in methods:
    print("MMSB")
    GMM = []
    times = 0
    for graph in tqdm(graphs):
      N, M = len(graph.nodes()), len(graph.edges())
      D = max(1, int(2 * M / N))
      C = max(1, int(N / D))
      p = 1.0
      mm_gen = MMSB(N, C, p)
      start_time = time.time()
      G = mm_gen.generate_graph()
      end_time = time.time()
      times += end_time - start_time
      GMM.append(G)
      if len(GMM) == NUM:
        break
    draw(GMM, args.dataset, "MM")
    eval(graphs, GMM)
    print("total times: %f" % (times / len(graphs)))
    del GMM

  if "Kronecker" in methods:
    print("Kronecker")
    GKro = []
    times = 0
    for graph in tqdm(graphs):
      start_time = time.time()
      N, M = len(graph.nodes()), len(graph.edges())
      D = max(1, int(2 * M / N))
      # print(N, D)
      # input()
      G = kronecker(N, D)
      end_time = time.time()
      times += end_time - start_time
      GKro.append(G)
      # nx.draw(G)
      # plt.show()
      # plt.close()
      if len(GKro) == NUM:
        break
    draw(GKro, args.dataset, "Kronecker")
    eval(graphs, GKro)
    print("total times: %f" % (times / len(graphs)))
    del GKro

  if "Ours" in methods:
    print("Ours")
    GOurs = []
    times = 0
    T1 = 0
    T2 = 0
    T3 = 0
    TN = 0
    TNM = 0
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
      maxlim = lim
      # ===========================================================
      G = graph
      MAXD = len(nx.degree_histogram(G)) - 1
      if 2 * M == N * (N - 1):
        G = nx.complete_graph(N)
        GOurs.append(G)
        continue

      # 2. pick (d+1, d), d = MAXD
      subs = [(MAXD + 1, MAXD)]
      sN, sM = N - MAXD - 1, M - MAXD - 1
      sumN, sumM = MAXD + 1, MAXD
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
      T1 += end_time - start_time

      # 3. build edges between nodes
      # 3.0 prepare
      start_time = time.time()
      entries = [[sub[1]] + [1 for _ in range(sub[1])] for sub in subs]

      # TODO: really need?
      # 3.1 make sure to connect
      # new faster method
      # base_plist = [len(entry) for entry in entries]
      # base_plist = [val / sum(base_plist) for val in base_plist]
      # all_plist = []
      # for i in range(len(entries)):
      #   temp = [val for val in base_plist]
      #   temp[i] = 0
      #   temp = [val / sum(temp) for val in temp]
      #   all_plist.append(temp)
      edges = []
      if len(subs) > 1:
        for i, sub in enumerate(subs):
          if i == 0:
            continue
          # ========================================================================
          # pick under MAXD
          # goi = [poisson(x + 1, D, maxlim) if x < MAXD else 0 for x in entries[i]]
          # pick without limit
          goi = [poisson(x + 1, D, maxlim) for x in entries[i]]
          if sum(goi) == 0:
            continue
          # pick without limit
          # goi[0] = 0 if sum(goi[1:]) != 0 else goi[0]
          goi = [val / sum(goi) for val in goi]
          # ========================================================================
          # pick without limit
          # goi = [poisson(x + 1, D) for x in entries[i]]
          # if sum(goi) == 0:
          #   continue
          # goi[0] = 0
          # goi = [val / sum(goi) for val in goi]
          # ========================================================================

          ll = random_sampler(len(goi), goi)

          # =======================================================================

          j = r.randint(0, i - 1)
          # ========================================================================
          # pick under MAXD
          # goi = [poisson(x + 1, D, maxlim) if x < MAXD else 0 for x in entries[i]]
          # pick without limit
          goj = [poisson(x + 1, D, maxlim) for x in entries[j]]
          if sum(goj) == 0:
            continue
          # pick without limit
          # goi[0] = 0 if sum(goi[1:]) != 0 else goi[0]
          goj = [val / sum(goj) for val in goj]
          # ========================================================================
          # pick without limit
          # goi = [poisson(x + 1, D) for x in entries[i]]
          # if sum(goi) == 0:
          #   continue
          # goi[0] = 0
          # goi = [val / sum(goi) for val in goi]
          # ========================================================================

          rr = random_sampler(len(goj), goj)

          # =======================================================================

          # nlist = []
          # plist = []
          # for subid, elist in enumerate(entries):
          #   if subid == i:
          #     continue
          #   psub = len(elist) / sumN
          #   sublist = []
          #   for eid, edeg in enumerate(elist):
          #     # pick under MAXD
          #     p = psub * (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
          #     # p = (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
          #     # pick without limit
          #     # p = psub * (poisson(edeg + 1, D, maxlim))
          #     nlist.append((subid, eid))
          #     sublist.append(p)
          #   # pick without limit
          #   # sublist[0] = 0 if sum(sublist[1:]) != 0 else sublist[0]
          #   plist.extend(sublist)
          # sumt = sum(plist)
          # plist = [p / sumt for p in plist]

          # v = random_sampler(len(nlist), p=plist)
          # v, rr = nlist[v][0], nlist[v][1]
          # =======================================================================
          # v = random_sampler(len(all_plist[i]), p=all_plist[i])
          # gov = [poisson(x + 1, D) if x < MAXD else 0 for x in entries[v]]
          # if sum(gov) == 0:
          #   continue
          # gov[0] = 0 if sum(gov[1:]) != 0 else gov[0]
          # gov = [val / sum(gov) for val in gov]
          # rr = random_sampler(len(gov), gov)
          # =======================================================================
          entries[i][ll] += 1
          entries[j][rr] += 1
          edges.append(((i, ll), (j, rr)))
          if len(edges) == sM:
            break
      end_time = time.time()
      times += end_time - start_time
      T2 += end_time - start_time

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
      for edge in edges:
        u = indice[edge[0][0]] + edge[0][1]
        v = indice[edge[1][0]] + edge[1][1]
        edge_index.append((u, v))
      G = nx.empty_graph()
      G.add_nodes_from(indice)
      G.add_edges_from(edge_index)
      pN, pM = len(G.nodes()), len(G.edges())
      # ===========================================================================

      # 3.2 basic entries
      res_edges = M - pM
      extra_edges = []
      # ====================================================================
      # outer update
      # nlist = []
      # plist = []
      # for sub, elist in enumerate(entries):
      #   psub = len(elist) / sumN
      #   sublist = []
      #   for eid, edeg in enumerate(elist):
      #     p = psub * (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
      #     # p = psub * (poisson(edeg, D))
      #     nlist.append((sub, eid))
      #     sublist.append(p)
      #   sublist[0] = 0 if sum(sublist[1:]) != 0 else sublist[0]
      #   plist.extend(sublist)
      # sumt = sum(plist)
      # print(plist)
      # plist = [p / sumt for p in plist]
      # print(plist)
      # # print(len(nlist), len(plist))
      # input()
      # # ====================================================================
      # edges = itertools.combinations(range(N), 2)    
      # cnt = 0   
      # start_time = time.time()
      # for edge in edges:
      #   u, v = edge[0], edge[1]
      #   if G.has_edge(u, v):
      #     continue
      #   p = plist[edge[0]] * plist[edge[1]]
      #   if np.random.random() < p:
      #     G.add_edge(*edge)
      #     u, uentry = nlist[u][0], nlist[u][1]
      #     v, ventry = nlist[v][0], nlist[v][1]
      #     entries[u][uentry] += 1
      #     entries[v][ventry] += 1
      #     cnt += 1
      #     if res_edges == cnt:
      #       break
      #     if res_edges // 2 == cnt:
      #       nlist = []
      #       plist = []
      #       for sub, elist in enumerate(entries):
      #         psub = len(elist) / sumN
      #         sublist = []
      #         for eid, edeg in enumerate(elist):
      #           p = psub * (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
      #           # p = psub * (poisson(edeg, D))
      #           nlist.append((sub, eid))
      #           sublist.append(p)
      #         sublist[0] = 0 if sum(sublist[1:]) != 0 else sublist[0]
      #         plist.extend(sublist)
      #       sumt = sum(plist)
      #       plist = [p / sumt for p in plist] 
      #       res_edges = cnt
      #       cnt = 0
               
      # end_time = time.time()
      # times += end_time - start_time

      start_time = time.time()
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
            p = psub * (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
            # p = (poisson(edeg + 1, D, maxlim) if edeg < MAXD else 0)
            # pick without limit
            # p = psub * (poisson(edeg + 1, D, maxlim))
            nlist.append((sub, eid))
            sublist.append(p)
          # pick without limit
          # sublist[0] = 0 if sum(sublist[1:]) != 0 else sublist[0]
          plist.extend(sublist)
        # ====================================================================
        sumt = sum(plist)
        if sumt == 0:
          break
        plist = [p / sumt for p in plist]
        asampler = Alias(plist)

        # V2 of picking
        eset = set()
        for _ in range(res_edges):
          # u, v = random_sampler(len(nlist), p=plist, size=2, replace=False)
          # u, v = asampler.sample(2)
          u = asampler.sample()
          v = asampler.sample()
          if u != v:
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
      end_time = time.time()
      times += end_time - start_time
      T3 += end_time - start_time

      G.add_edges_from(extra_edges)
      GOurs.append(G)
      if len(GOurs) == NUM:
        break
    # for i, G in enumerate(GOurs):
    #   # colors = ["#B63D3D" if val in indice else "#557AA4" for val in range(G.number_of_nodes())]
    #   colors = ["#C66218" if node in indice else "#004285" for node in G.nodes()]
    #   nx.draw(G, pos=nx.spring_layout(G), node_color=colors, node_size=100)
    #   plt.savefig(".\\visual\\%s\\%d_ours.png" % (args.dataset, i))
    #   # plt.show()
    #   plt.close()
    eval(graphs, GOurs)
    print("total times: %f" % (times / len(graphs)))
    print("T1 times: %f" % (T1 / len(graphs)))
    print("T2 times: %f" % (T2 / len(graphs)))
    print("T3 times: %f" % (T3 / len(graphs)))
    # print("TN times: %f" % (TN / len(graphs)))
    # print("TNM times: %f" % (TNM / len(graphs)))
    del GOurs

  del graphs