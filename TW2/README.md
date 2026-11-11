# ID2222 Homework 2

这个项目实现了 A-Priori 算法，用于从销售交易数据中找出频繁项集。
程序也包含关联规则生成，供作业的可选加分部分使用。

## 文件说明

- `apriori.py` 保存算法、命令行参数和结果导出逻辑。
- `apriori_notebook.ipynb` 用更适合学习的方式解释算法并展示实验。
- `test_apriori.py` 使用小数据集检查支持度、剪枝和关联规则。
- `run.bat` 在 Windows 中运行一个完整示例。
- `results/` 保存 CSV 和 JSON 实验结果。
- `IMPLEMENTATION_PLAN.md` 记录完整设计方案。

数据集放在项目上一级目录的 `T10I4D100K.dat`，也就是：

```text
C:\Users\goahe\Desktop\ID2222\Coding\T10I4D100K.dat
```

最终提交时不要把数据集文件放进 zip。报告中需要保留数据集来源和下载链接。

## 环境要求

只需要 Python 3.10 或更新版本。本项目只使用 Python 标准库，不需要安装第三方包。

## 先运行小数据集测试

在 `TW2` 目录中执行：

```text
python test_apriori.py
```

如果实现正确，程序会打印：

```text
All small-dataset A-Priori checks passed.
```

## 运行必做部分

`support` 是交易数量。例如 `1000` 表示一个 itemset 至少出现在 1000 条交易中。

```text
python apriori.py --dataset ..\T10I4D100K.dat --support 1000
```

## 运行可选的关联规则部分

`confidence` 使用 0 到 1 之间的小数：

```text
python apriori.py --dataset ..\T10I4D100K.dat --support 1000 --confidence 0.6
```

## 输出文件

默认结果会写到 `results` 目录：

- `frequent_itemsets.csv` 保存项集大小、项集内容和支持度。
- `association_rules.csv` 在提供 `--confidence` 时生成。
- `run_summary.json` 保存交易数、参数、每一层耗时和总耗时。

可以使用其他目录：

```text
python apriori.py --dataset ..\T10I4D100K.dat --support 1000 --output-dir results_support_1000
```

## Notebook

打开 `apriori_notebook.ipynb` 后，按顺序运行所有单元格。Notebook 会自动尝试当前目录、`TW2` 子目录和当前目录的上级 `TW2` 目录。Notebook 会：

1. 解释支持度和置信度。
2. 用小数据集进行断言测试。
3. 使用完整数据集运行 A-Priori。
4. 展示频繁项集数量和部分结果。
5. 记录每一层和总运行时间。
6. 在指定置信度后展示关联规则。

Notebook 中的 `MIN_SUPPORT` 和 `MIN_CONFIDENCE` 是示例参数。作业页面没有给出固定值，如果课程公告或 TA 给出指定值，需要替换成课程要求。

## 设计说明

程序使用 A-Priori 的向下封闭性质。上一层所有子集都频繁时，候选项集才会进入计数阶段。交易使用集合表示，所以一条交易中重复出现的项目不会被重复计算。

程序输出前会对 itemset 和规则排序，使结果可以复现，方便报告和现场演示。

## 提交前检查

- 运行 `python test_apriori.py`。
- 运行完整数据集并保存运行时间。
- 运行 Notebook 的所有单元格。
- 检查报告中包含构建方法、运行命令、参数、结果和运行时间。
- 在报告中写数据集获取方式和链接。
- 添加 AI Disclosure statement。
- 打包源代码、运行脚本、Notebook 和报告。
- 不要把 `T10I4D100K.dat` 放入提交压缩包。
