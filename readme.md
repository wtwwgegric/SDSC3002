project/
├── src/                 # 核心算法模块（.py）
│   ├── preprocess.py    # 数据加载与预处理
│   ├── baseline.py      # 暴力相似度计算
│   ├── lsh.py           # MinHash + LSH 实现
│   └── utils.py         # 辅助函数
├── experiments/         # 实验脚本（可选，或用 ipynb）
│   ├── run_baseline.py  # 运行基线并记录时间
│   └── run_lsh.py       # 运行不同参数组合
├── notebooks/           # Jupyter notebooks（实验分析）
│   ├── 1_data_explore.ipynb
│   ├── 2_lsh_parameter_tuning.ipynb
│   └── 3_results_viz.ipynb
├── data/                # 数据集（或软链接）
├── requirements.txt     # 依赖包列表
├── README.md            # 运行说明
└── report.pdf           # 最终报告