# ID2222 Homework 2

这个项目实现了 A-Priori 算法，用于从销售交易数据中找出频繁项集。
程序也包含关联规则生成，对应作业的可选加分部分。

## 文件说明

- `apriori.py` 保存算法、命令行参数和结果导出逻辑。
- `Homework2_Analysis.ipynb` 是报告，解释算法并展示实验结果和运行时间。
- `test_apriori.py` 用小数据集和暴力枚举检查支持度、剪枝和关联规则。
- `run.bat` 在 Windows 中运行一个完整示例，`run.sh` 用于 macOS 和 Linux。
- `results/` 保存 CSV 和 JSON 实验结果。
- `IMPLEMENTATION_PLAN.md` 记录设计方案和验证记录。

## 数据集

使用课程提供的 T10I4D100K 数据集。它包含 100,000 条生成的交易，每行是一条交易，项目是用空格分隔的数字编号。

- 课程 Canvas 链接，需要登录：<https://canvas.kth.se/courses/65339/files/10847690/download?wrap=1>
- 内容相同的公开镜像：<https://raw.githubusercontent.com/yiweig/Apriori/master/T10I4D100K.dat>

把下载的文件命名为 `T10I4D100K.dat`，放在 `TW2` 的上一级目录：

```text
ID2222/
├── T10I4D100K.dat
└── TW2/
```

macOS / Linux 可以在 `TW2` 的上一级目录执行：

```sh
curl -L -o T10I4D100K.dat "https://raw.githubusercontent.com/yiweig/Apriori/master/T10I4D100K.dat"
```

文件大小是 4,022,055 字节，共 100,000 行。最终提交时不要把数据集文件放进 zip。

程序也可以读取其他交易数据集，只要每行是一条交易、项目之间用空白字符分隔。

## 环境要求

只需要 Python 3.10 或更新版本。`apriori.py` 和测试只使用 Python 标准库，不需要安装第三方包，也没有构建步骤。运行 Notebook 另外需要 `notebook` 和 `matplotlib`：

```sh
python -m pip install notebook matplotlib
```

下面的命令在 Windows 上使用 `python`，在 macOS / Linux 上通常是 `python3`。Windows 的路径可以写成 `..\T10I4D100K.dat`。

## 先运行小数据集测试

在 `TW2` 目录中执行：

```sh
python test_apriori.py
```

测试不需要数据集。如果实现正确，程序会打印：

```text
All small-dataset A-Priori checks passed.
```

测试覆盖：

- 人工可算的四条交易上的支持度和规则
- 候选生成的 join 和 prune 两步
- 支持度计数的两条路径
- 200 个随机小数据集上，频繁项集和关联规则与暴力枚举完全一致
- 空行、重复项目、空文件、不存在的文件和非法参数

## 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--dataset` | 必填 | 交易文件路径，每行一条交易，项目用空白字符分隔 |
| `--support` | 必填 | 最小支持度 s，是交易数量，必须是正整数 |
| `--confidence` | 不生成规则 | 最小置信度 c，0 到 1 之间的小数。提供时才生成关联规则 |
| `--output-dir` | `results` | 保存 CSV 和 JSON 结果的目录 |

## 运行必做部分

`support` 是交易数量。例如 `1000` 表示一个 itemset 至少出现在 1000 条交易中。

```sh
python apriori.py --dataset ../T10I4D100K.dat --support 1000
```

## 运行可选的关联规则部分

`confidence` 使用 0 到 1 之间的小数：

```sh
python apriori.py --dataset ../T10I4D100K.dat --support 1000 --confidence 0.6
```

运行脚本 `run.sh` 和 `run.bat` 执行的就是这条命令。

## 输出文件

程序会在终端打印交易数、每一层的候选数量、频繁项集数量和耗时。默认结果会写到 `results` 目录：

- `frequent_itemsets.csv` 保存项集大小、项集内容和支持度。
- `association_rules.csv` 在提供 `--confidence` 时生成，保存前件、后件、支持度和置信度。
- `run_summary.json` 保存交易数、参数、每一层耗时和总耗时。

可以使用其他目录：

```sh
python apriori.py --dataset ../T10I4D100K.dat --support 1000 --output-dir results_support_1000
```

## Notebook

在 `TW2` 目录运行 `jupyter notebook`，打开 `Homework2_Analysis.ipynb` 后按顺序运行所有单元格。Notebook 会自动尝试当前目录、`TW2` 子目录和当前目录的上级 `TW2` 目录。Notebook 会：

1. 解释支持度和置信度。
2. 用小数据集进行断言测试。
3. 使用完整数据集运行 A-Priori。
4. 展示频繁项集数量和部分结果。
5. 记录每一层和总运行时间。
6. 生成并展示关联规则。
7. 比较不同支持度阈值下的频繁项集数量和运行时间。

Notebook 中的 `MIN_SUPPORT = 1000` 和 `MIN_CONFIDENCE = 0.6` 是本次实验选择的参数。作业页面没有给出固定值。

## 设计说明

程序使用 A-Priori 的向下封闭性质：一个 itemset 频繁，它的所有子集也必须频繁。

- 候选生成使用 Agrawal 和 Srikant 论文中的 apriori-gen。上一层的频繁项集写成有序元组，前 `k-2` 个项目相同的两个项集合并成一个大小为 `k` 的候选，再删除含有不频繁子集的候选。
- 支持度计数每层扫描一遍交易。每条交易先缩减到出现在候选项集中的项目，再枚举大小为 `k` 的组合并在候选的哈希表中查找。组合数比候选数多时，改为逐个检查候选是否是交易的子集。
- 交易使用集合表示，所以一条交易中重复出现的项目不会被重复计算。
- 关联规则对每个大小至少为 2 的频繁项集枚举所有非空真子集作为前件，置信度是 `support(X ∪ Y) / support(X)`。规则的支持度就是频繁项集的支持度，所以一定不低于 s。

程序输出前会对 itemset 和规则排序，项目编号按数值顺序排列，使结果可以复现，方便报告和现场演示。

## 提交前检查

- 运行 `python test_apriori.py`。
- 运行完整数据集并保存运行时间。
- 运行 Notebook 的所有单元格。
- 打包源代码、运行脚本、README 和 Notebook。Notebook 已包含报告内容、数据集链接和 AI Disclosure statement。
- 不要把 `T10I4D100K.dat` 放入提交压缩包。
