PY ?= .venv/bin/python

.PHONY: setup test trace stage1 stage1-quick quick full outcomes standoff plots report clean

setup:            ## create the virtualenv and install dependencies
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

test:             ## Stage 0 deliverable: unit tests (incl. hand-checked ring traces)
	$(PY) -m pytest -q

trace:            ## print a hand-checkable trace of the three-agent ring
	$(PY) experiments/stage0_ring_trace.py

stage1-quick:     ## Stage 1 on a reduced grid (~7 min)
	$(PY) experiments/stage1_two_agent.py --quick

stage1:           ## Stage 1 on the full (R/K, D, rho) grid
	$(PY) experiments/stage1_two_agent.py

quick:            ## Stages 2-4 + baselines on a reduced grid, 3 seeds
	$(PY) experiments/run_grid.py --preset all --quick

full:             ## Stages 2-4 + baselines on the proposal's full grid, 10 seeds (hours)
	$(PY) experiments/run_grid.py --preset all

outcomes:         ## rho = 0 cells again, with the stall metric (identical seeds; ~1 h)
	$(PY) experiments/run_grid.py --preset all --rho 0 --out results/outcomes

standoff:         ## every method on the two-agent game of Stage 1 (criterion 1; ~1 h)
	$(PY) experiments/run_grid.py --topologies standoff --methods brd iql fp sql spe always-request bankers \
	  resource-ordering exp-backoff centralized --out results/standoff

plots:            ## figures from whatever results exist
	$(PY) experiments/make_plots.py --stage1 results/stage1 --grid $(wildcard results/all*) --out results/figures

report:           ## every table and figure of report/main.tex (+ report/numbers.txt), then the PDF if LaTeX is installed
	$(PY) experiments/make_report_assets.py
	-cd report && (command -v tectonic >/dev/null && tectonic -X compile main.tex \
	  || (pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex) >/dev/null)

clean:
	rm -rf results/*_quick results/figures .pytest_cache
