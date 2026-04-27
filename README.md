# MCCG
**TL;DR:** Author Name Disambiguation (AND) suffers from noisy heuristic graphs and the lack of joint representation–clustering optimization. We propose MCCG, a multi-view contrastive and cluster-guided framework that strengthens robustness to graph noise and unifies representation learning with clustering-based disambiguation. By leveraging homogeneous paper clustering and dynamically refining reliable cluster labels, MCCG learns more discriminative embeddings. Experiments on four benchmark datasets show that MCCG consistently outperforms state-of-the-art methods.

## Overview
<img width="1419" alt="image" src="https://github.com/user-attachments/assets/f5af2fdf-c05d-42a2-8277-e00db5fbb7c2" />

## Dataset
[AMiner-AND](https://www.aminer.cn/na-data) 
[WhoisWho-v1 and WhoisWho-v2](https://www.aminer.cn/billboard/whoiswho)
[LAGOS-AND](https://zenodo.org/records/7313380)

When using the LAGOS-AND dataset, we follow the approach in “Name Disambiguation in AMiner: Clustering, Maintenance, and Human in the Loop” by filtering out ambiguous authors with fewer than five publications. This may result in some ambiguous names having no associated authors, which are subsequently removed. Other datasets already exhibit this property at creation time, and we adopt this strategy considering the suitability of LAGOS-AND for the author name disambiguation task.

## Quick Start
```python
python main.py
```

## Citation
```
@article{ye2025multi,
  title={Multi-view Contrastive and Cluster-Guided Learning for Author Name Disambiguation},
  author={Ye, Fan and Xia, Zong and Ling, Zhaolong and Wu, Le},
  journal={Expert Systems with Applications},
  pages={128324},
  year={2025},
  publisher={Elsevier}
}
```
