PY   := .venv/bin/python
QRUN := .venv/bin/qrun

# 数据截止日固定, 保证实验可复现 (如需更新数据改这里或 make fetch DATA_END=YYYYMMDD)
DATA_END ?= 20260821

# mlflow>=3(pyqlib 传递依赖)将本地文件后端设为维护模式, 而 qrun/qlib 依赖 ./mlruns
# 文件存储 —— 用 mlflow 官方 opt-out 开关放行(替代方案: requirements 锁 mlflow<3)
MLFLOW_ENV ?= MLFLOW_ALLOW_FILE_STORE=true

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
		--data_path data/csv_raw \
		--qlib_dir data/qlib_data/cn_data \
		--include_fields open,close,high,low,volume,factor,vwap
	@grep -v "^SH000300" data/qlib_data/cn_data/instruments/all.txt > data/qlib_data/cn_data/instruments/csi300.txt
	@echo "股票池: csi300.txt 共 $$(wc -l < data/qlib_data/cn_data/instruments/csi300.txt | tr -d ' ') 只(已剔除基准指数)"

data: fetch dump

train:
	@mkdir -p reports
	bash -c 'set -o pipefail; $(MLFLOW_ENV) $(QRUN) config/workflow_lightgbm.yaml 2>&1 | tee reports/qrun_output.log'

analyze:
	$(MLFLOW_ENV) $(PY) scripts/analyze_recorder.py

all: setup data train analyze

clean:
	rm -rf data/csv_raw data/qlib_data mlruns reports/*
