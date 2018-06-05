.PHONY: init app test

init:
	pip install -r requirements.txt

app:
	PYTHONPATH="${PYTHONPATH}:./src" python ./src/app.py

test:
	PYTHONPATH="${PYTHONPATH}:./src" pytest --cov-report=term --cov=src
