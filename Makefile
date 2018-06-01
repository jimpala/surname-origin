.PHONY: init test

init:
	pip install -r requirements.txt

test:
	PYTHONPATH="${PYTHONPATH}:./src" pytest --cov-report=term --cov=src
