#!/usr/bin/env sh
# macOS / Linux 运行脚本。需要先按 README 下载数据集。
# 额外参数会原样传给 main.py，例如：./run.sh --threshold 0.5
cd "$(dirname "$0")" || exit 1
python3 main.py --sizes 10 100 1000 --lsh "$@"
