PY   := .venv/bin/python
QRUN := .venv/bin/qrun

# 数据截止日固定, 保证实验可复现 (如需更新数据改这里或 make fetch DATA_END=YYYYMMDD)
DATA_END ?= 20260821

# mlflow>=3(pyqlib 传递依赖)将本地文件后端设为维护模式, 而 qrun/qlib 依赖 ./mlruns
# 文件存储 —— 用 mlflow 官方 opt-out 开关放行(替代方案: requirements 锁 mlflow<3)
MLFLOW_ENV ?= MLFLOW_ALLOW_FILE_STORE=true

.PHONY: setup fetch dump data train train10w signal analyze analyze10w paper paper-status all clean

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

# 10万元模拟盘回测: 模型/数据与 train 完全一致, 仅账户规模与持仓数不同
# 写入独立实验 workflow_10w(yaml 顶层 experiment_name), 与主实验产物隔离
train10w:
	@mkdir -p reports
	bash -c 'set -o pipefail; $(MLFLOW_ENV) $(QRUN) config/workflow_lightgbm_10w.yaml 2>&1 | tee reports/qrun_output_10w.log'

signal:
	$(MLFLOW_ENV) $(PY) scripts/today_signal.py

# 每日自动模拟盘: 收盘后增量抓数→打分→按真实收盘价模拟成交→逐日记账
# 首次使用先: .venv/bin/python scripts/paper_trade.py --init
paper:
	$(MLFLOW_ENV) $(PY) scripts/paper_trade.py

# 只打印模拟盘现状, 不入账不抓数
paper-status:
	$(MLFLOW_ENV) $(PY) scripts/paper_trade.py --status

analyze:
	$(MLFLOW_ENV) $(PY) scripts/analyze_recorder.py

# 10w 实验独立导出(experiment_name=workflow_10w, 与主实验 recorder 互不混淆)
# 注意: 两版导出仍写同一 reports/ 目录, 需要并存时先备份
analyze10w:
	$(MLFLOW_ENV) $(PY) scripts/analyze_recorder.py --experiment workflow_10w

all: setup data train analyze

clean:
	rm -rf data/csv_raw data/qlib_data mlruns reports/*
