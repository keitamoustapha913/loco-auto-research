.PHONY: install test start status stop
install:
	uv sync --extra all
test:
	uv run --extra all pytest
start:
	uv run --extra all loco-lab start
status:
	uv run --extra all loco-lab status
stop:
	uv run --extra all loco-lab stop
