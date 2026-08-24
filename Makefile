PY   := .venv/bin/python
QRUN := .venv/bin/qrun

# 数据截止日固定, 保证实验可复现 (如需更新数据改这里或 make fetch DATA_END=YYYYMMDD)
DATA_END ?= 20260821

.PHONY: setup fetch dump data train analyze all clean

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt
	@$(PY) -c "import qlib, akshare, lightgbm" 2>/dev/null || \
		{ echo "import 失败: macOS 上 LightGBM 需要系统级 OpenMP, 请先执行: brew install libomp"; exit 1; }

fetch:
	$(PY) scripts/fetch_data.py --start 20200901 --end $(DATA_END)

dump:
	$(PY) scripts/dump_bin.py dump_all \
		--csv_path data/csv_raw \
		--qlib_dir data/qlib_data/cn_data \
		--include_fields open,close,high,low,volume,factor,vwap \
		--exclude_fields symbol

data: fetch dump

train:
	@mkdir -p reports
	bash -c 'set -o pipefail; $(QRUN) config/workflow_lightgbm.yaml 2>&1 | tee reports/qrun_output.log'

analyze:
	$(PY) scripts/analyze_recorder.py

all: setup data train analyze

clean:
	rm -rf data/csv_raw data/qlib_data mlruns reports/*
