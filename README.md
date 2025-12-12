# A Hierarchical Scale-free Graph Generator under Limited Resources

Code for *A Hierarchical Scale-free Graph Generator under Limited Resources* (under review).

A non-learned hierarchical scale-free graph generator under limited resources, since the training data may sometimes not available due to various concerns like privacy, security, copyright, legal issues, etc.
It contains three stages: parsing, connecting, and rebuilding.
It can generate graphs under limited generation resources closer to real-world distributions with only a few parameters, including the basic graph statistics and a representitive observation of scale-free property (e.g., Poisson distribution).
Experiments on 12 datasets in three categories demonstrates the advantages of our work.

## Requirement
We recommend using a conda environment.

**For conda** set up, you could run the code:

```bash
conda env create -f environment.yml
conda activate graph-generation
```

**For pip** set up, you could also run this code:

```powershell
pip install -r requirements.txt
```

## Quick Start
For quick start, you could run the code:
```powershell
python nxgen.py
  --dataset <dataset name>
  --method <method name>
  --num <the number of graphs>
```
For example, 
```powershell
python nxgen.py --dataset NCI1 --method Ours --num 10
```

## Detailed Settings
- dataset: supported dataset names
  - bioinformatics & molecules: `ENZYMES`, `MUTAG`, `NCI1`, `PROTEINS`
  - social networks: `deezer_ego_nets`, `IMDB-BINARY`, `IMDB-MULTI`, `REDDIT-BINARY`
  - synthetic: `CLUS`, `EGO`, `GRID`, `TREE`
- method: supported baselines, `ER`, `BA`, `WS`, `MMSB`, `Kronecker`, `Ours`
  - use `all` to compare methods in a single run
- num (optional): the number of graphs

## Result Format
Results are in format:
```txt
<method name>
...
Time computing degree mmd: ...
<the degree MMD>
Time computing clustering mmd: ...
<the clustering MMD>
<the orbit MMD>
total times: <the total runtime>
T1 times: <the parsing stage runtime>
T2 times: <the connecting stage runtime>
T3 times: <the rebuilding stage runtime>
```

## Acknowledgements
We express our gratitude for the powerful graph generation works, Erdos-Renyi (ER), Barabasi-Albert (BA), Watts-Strogatz (WS), Mixed-Membership Stochasitc Block (MMSB), and Kronecker.

## Citation
Please consider citing our work if you find it helpful:
```bibtex
@article{DBLP:journals/corr/abs-2411-13888,
  author       = {Xiaorui Qi and
                  Yanlong Wen and
                  Xiaojie Yuan},
  title        = {A Hierarchical Scale-free Graph Generator under Limited Resources},
  journal      = {CoRR},
  volume       = {abs/2411.13888},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2411.13888},
  doi          = {10.48550/ARXIV.2411.13888},
  eprinttype    = {arXiv},
  eprint       = {2411.13888},
  timestamp    = {Wed, 01 Jan 2025 13:20:22 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2411-13888.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
**Note: the bib above is an early preprint, the new version is coming soon.**

## FAQ
- Q1: ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject

A1: Often caused by the version conflict of `numpy`. Please check the **Requirement** section to set up properly.