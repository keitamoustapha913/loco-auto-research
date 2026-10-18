.PHONY: install test demo mlflow dashboard status worker
install:
	python -m pip install -e '.[all]'
test:
	python -m pytest -q
demo:
	python -m scripts.demo_run
mlflow:
	bash scripts/start_mlflow.sh
dashboard:
	streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
status:
	python -m orchestrator.controller status
worker:
	python -m orchestrator.controller worker-once
