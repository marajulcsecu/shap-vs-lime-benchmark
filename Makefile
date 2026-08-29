PY := .venv/bin/python
PROFILE ?= QUICK
export OMP_NUM_THREADS = 1
export PYTHONHASHSEED = 42
export PYTHONPATH = src

.PHONY: setup data test smoke full pipeline all figures tables paper report slides docs clean

setup:
	uv venv --python 3.12
	uv pip install --python .venv/bin/python -r requirements.lock.txt

data:
	$(PY) -m xaibench.data.fetch

test:
	.venv/bin/pytest -q

stage1:
	XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage1_performance
stage2:
	XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage2_explain
stage3:
	XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage3_metrics
stage4:
	XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage4_stats
stage5:
	XAIB_PROFILE=$(PROFILE) $(PY) -m xaibench.pipeline.stage5_outputs

pipeline: stage1 stage2 stage3 stage4 stage5

smoke:
	$(MAKE) data && $(MAKE) test && $(MAKE) pipeline PROFILE=QUICK
full:
	$(MAKE) data && $(MAKE) pipeline PROFILE=FULL

figures tables: stage5

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
slides:
	cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
docs: paper report slides

all: full docs

clean:
	rm -rf results figures tables
	-cd paper && latexmk -C
	-cd report && latexmk -C
	-cd slides && latexmk -C
