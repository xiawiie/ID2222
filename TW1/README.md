# Homework 1: Finding Similar Items

这个项目使用 Twenty Newsgroups 的 mini 数据集，实现作业要求的文本相似度流程。

项目覆盖：

1. Shingling
2. 直接 Jaccard similarity
3. MinHashing
4. MinHash signature similarity
5. 可选的 LSH 候选搜索
6. 10、100、1,000 篇文档的规模实验

## 目录结构

~~~~text
TW1/
├── similarity.py
├── main.py
├── Homework1_Analysis.ipynb
├── test_similarity.py
├── README.md
├── run.ps1
└── run.bat
~~~~

数据集不复制到项目目录，也不要放进最终提交 ZIP。

## 数据集

默认读取：

~~~~text
..\twenty_newsgroups\mini\mini_newsgroups
~~~~

当前电脑上的实际路径是：

~~~~text
C:\Users\goahe\Desktop\ID2222\Coding\twenty_newsgroups\mini\mini_newsgroups
~~~~

mini 数据集包含 20 个新闻组，每个新闻组 100 篇文档，共 2,000 篇。

每篇文档是一个 Latin-1 编码的 Usenet 新闻组消息。程序默认保留 Subject 和正文，并去掉 Message-ID、日期等动态邮件头。

## 环境

- Python 3.10 或更高版本
- Jupyter Notebook
- matplotlib，用于 Notebook 绘图
- 核心算法只使用 Python 标准库

如果本地还没有 Jupyter 或 matplotlib，可以在已有 Python 环境中安装：

~~~~powershell
python -m pip install notebook matplotlib
~~~~

## 命令行运行

在 PowerShell 中进入项目目录：

~~~~powershell
cd C:\Users\goahe\Desktop\ID2222\Coding\TW1
~~~~

运行完整规模实验：

~~~~powershell
python main.py --sizes 10 100 1000 --lsh
~~~~

这个命令默认使用：

- 字符 shingle 长度 k=5
- MinHash signature 长度 100
- Jaccard 相似度阈值 0.3
- 20 个 LSH bands
- 每个 band 5 行
- 保留 Subject 和正文

结果会保存到：

~~~~text
TW1/results/experiment_summary.csv
TW1/results/top_pairs.csv
~~~~

只运行两个主题的算法演示：

~~~~powershell
python main.py --categories comp.graphics comp.os.ms-windows.misc --sizes 10 --lsh
~~~~

只使用正文：

~~~~powershell
python main.py --body-only --sizes 10 100
~~~~

改变参数：

~~~~powershell
python main.py --k 10 --num-hashes 200 --threshold 0.4 --sizes 10 100 1000 --lsh
~~~~

## 运行自检

~~~~powershell
python -m unittest -v
~~~~

自检覆盖：

- shingle 去重
- Jaccard 计算
- MinHash signature 长度
- 相同 signature 的相似度
- Subject 和正文的预处理
- LSH 候选对
- LSH 参数完整性检查

## 运行 Jupyter Notebook

在 TW1 目录运行：

~~~~powershell
jupyter notebook
~~~~

然后打开：

~~~~text
Homework1_Analysis.ipynb
~~~~

Notebook 会展示：

- 数据集结构
- 一篇文档的预处理结果
- shingle 哈希值
- exact Jaccard
- MinHash estimate
- MinHash 误差
- LSH 候选对和召回率
- 10、100、1,000 篇文档的运行时间
- k=5 和 k=10 的参数对比
- signature length 100 和 200 的参数对比

## 算法和代码的对应关系

| 作业概念 | 代码位置 |
|---|---|
| Shingling | similarity.py 的 Shingling |
| CompareSets | similarity.py 的 CompareSets |
| MinHashing | similarity.py 的 MinHashing |
| CompareSignatures | similarity.py 的 CompareSignatures |
| LSH | similarity.py 的 LSH |
| 实验入口 | main.py |
| 报告和图表 | Homework1_Analysis.ipynb |

## 结果解释

experiment_summary.csv 主要包含：

- documents
- pairs
- similar_pairs
- lsh_candidates
- lsh_true_positives
- lsh_recall
- shingling_seconds
- minhash_seconds
- pair_comparison_seconds
- lsh_seconds
- total_seconds

直接比较的文档对数量为：

~~~~text
n * (n - 1) / 2
~~~~

因此文档数量增加时，pair comparison 通常会成为主要耗时。

## 作业提交注意事项

最终 ZIP 应包含：

- 源代码
- 运行脚本
- README
- Jupyter Notebook
- 测试代码
- Notebook 生成的实验结果或图表
- AI Disclosure statement

不要包含：

- twenty_newsgroups 数据集
- twenty_newsgroups.zip
- __pycache__

如果 Notebook 已经包含完整报告内容，可以按照作业说明不再单独提交 PDF。
