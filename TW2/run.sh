#!/usr/bin/env sh
# macOS / Linux 运行脚本。需要先按 README 把 T10I4D100K.dat 放到 TW2 的上一级目录。
# support 和 confidence 是示例值，可以直接修改。
cd "$(dirname "$0")" || exit 1
python3 apriori.py --dataset "../T10I4D100K.dat" --support 1000 --confidence 0.6 --output-dir results
