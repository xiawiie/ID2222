# Homework 1: Finding Similar Items

这个项目使用 Twenty Newsgroups 的 mini 数据集，实现作业要求的文本相似度流程。

项目覆盖：

1. Shingling
2. 直接 Jaccard similarity
3. MinHashing
4. MinHash signature similarity
5. 10、100、1,000 篇文档的规模实验

## 目录结构

~~~~text
TW1/
├── similarity.py              核心算法
├── main.py                    命令行实验入口
├── Homework1_Analysis.ipynb   报告和图表
├── test_similarity.py         单元测试
├── README.md
├── run.sh                     macOS / Linux 运行脚本
├── run.ps1                    Windows PowerShell 运行脚本
├── run.bat                    Windows cmd 运行脚本
└── results/                   实验结果 CSV
~~~~

数据集不放在 TW1 目录里，也不要放进最终提交 ZIP。

## 数据集

使用 UCI Machine Learning Repository 的 Twenty Newsgroups 数据集：

- 数据集页面：<https://archive.ics.uci.edu/dataset/113/twenty+newsgroups>
- 直接下载：<https://archive.ics.uci.edu/static/public/113/twenty+newsgroups.zip>

下载得到的 ZIP 里有 `mini_newsgroups.tar.gz`。把它解压到 TW1 的上一级目录，得到下面的结构：

~~~~text
ID2222/
├── TW1/
└── twenty_newsgroups/
    └── mini/
        └── mini_newsgroups/
            ├── alt.atheism/
            ├── comp.graphics/
            └── ...
~~~~

macOS / Linux 可以在 TW1 的上一级目录执行：

~~~~sh
curl -L -o twenty_newsgroups.zip "https://archive.ics.uci.edu/static/public/113/twenty+newsgroups.zip"
unzip twenty_newsgroups.zip mini_newsgroups.tar.gz
mkdir -p twenty_newsgroups/mini
tar -xzf mini_newsgroups.tar.gz -C twenty_newsgroups/mini
~~~~

如果数据集放在其他位置，运行时用 `--data-dir` 指定 `mini_newsgroups` 目录。

mini 数据集包含 20 个新闻组，每个新闻组 100 篇文档，共 2,000 篇。

每篇文档是一个 Latin-1 编码的 Usenet 新闻组消息。程序默认保留 Subject 和正文，并去掉 Message-ID、日期等动态邮件头。

## 环境

- Python 3.10 或更高版本
- 核心算法和命令行程序只使用 Python 标准库，不需要构建步骤
- Jupyter Notebook 和 matplotlib 只在运行 Notebook 时需要

如果本地还没有 Jupyter 或 matplotlib，可以在已有 Python 环境中安装：

~~~~sh
python -m pip install notebook matplotlib
~~~~

## 命令行运行

先进入 TW1 目录。下面的命令在 Windows 上使用 `python`，在 macOS / Linux 上通常是 `python3`。

运行完整规模实验：

~~~~sh
python main.py --sizes 10 100 1000
~~~~

也可以直接使用运行脚本，效果相同：

~~~~text
./run.sh        macOS / Linux
.\run.ps1       Windows PowerShell
run.bat         Windows cmd
~~~~

### 参数和默认值

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data-dir` | `../twenty_newsgroups/mini/mini_newsgroups` | 数据集目录，每个子目录是一个新闻组 |
| `--categories` | 全部新闻组 | 只读取指定的新闻组 |
| `--sizes` | `10 100 1000` | 要测试的文档数量，可以给多个 |
| `--k` | `5` | 字符 shingle 的长度 |
| `--num-hashes` | `100` | MinHash signature 的长度 |
| `--threshold` | `0.3` | 判断文档相似的 Jaccard 阈值 s |
| `--body-only` | 关闭 | 只使用正文，不使用 Subject |
| `--output-dir` | `results` | 保存 CSV 结果的目录 |

结果会保存到：

~~~~text
TW1/results/experiment_summary.csv
TW1/results/top_pairs.csv
~~~~

### 其他示例

只运行两个主题的算法演示：

~~~~sh
python main.py --categories comp.graphics comp.os.ms-windows.misc --sizes 10
~~~~

只使用正文：

~~~~sh
python main.py --body-only --sizes 10 100
~~~~

改变参数：

~~~~sh
python main.py --k 10 --num-hashes 200 --threshold 0.4 --sizes 10 100 1000
~~~~

## 运行自检

~~~~sh
python -m unittest -v
~~~~

自检不需要数据集，覆盖：

- shingle 去重和有序集合表示
- Jaccard 计算，包括空集合和不相交集合
- MinHash signature 长度、确定性，以及估计值接近 exact Jaccard
- signature 相似度的计算
- Subject 和正文的预处理

## 运行 Jupyter Notebook

在 TW1 目录运行：

~~~~sh
jupyter notebook
~~~~

然后打开 `Homework1_Analysis.ipynb`，按顺序运行所有单元格。在本项目的测试电脑上完整运行约需 1 分钟。

Notebook 会展示：

- 数据集结构和获取方式
- 一篇文档的预处理结果
- shingle 哈希值
- exact Jaccard
- MinHash estimate 和误差
- 10、100、1,000 篇文档的运行时间
- k=5 和 k=10 的参数对比
- signature length 100 和 200 的参数对比

## 算法和代码的对应关系

| 作业概念 | 代码位置 |
|---|---|
| Shingling | similarity.py 的 `Shingling` |
| CompareSets | similarity.py 的 `CompareSets` |
| MinHashing | similarity.py 的 `MinHashing` |
| CompareSignatures | similarity.py 的 `CompareSignatures` |
| 实验入口 | main.py |
| 报告和图表 | Homework1_Analysis.ipynb |

## 结果解释

experiment_summary.csv 每行对应一个文档规模，主要包含：

| 列 | 含义 |
|---|---|
| `documents` | 文档数量 |
| `pairs` | 文档对数量，等于 n(n-1)/2 |
| `similar_pairs` | exact Jaccard 不低于阈值的文档对数量，是准确基准 |
| `shingling_seconds` 等 | 各阶段和总运行时间，单位是秒 |

直接比较全部文档对的数量是 n(n-1)/2，随文档数量平方增长，文档数量增加时它会成为主要耗时。

## 作业提交注意事项

最终 ZIP 应包含：

- 源代码
- 运行脚本
- README
- Jupyter Notebook，其中包含报告内容和 AI Disclosure statement
- 测试代码
- results 目录中的实验结果

不要包含：

- twenty_newsgroups 数据集
- twenty_newsgroups.zip
- `__pycache__`

Notebook 已经包含完整报告内容，可以按照作业说明不再单独提交 PDF。
