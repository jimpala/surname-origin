init:
	pip install -r requirements.txt

test:
	export PYTHONPATH="${PYTHONPATH}:./src"
	pytest tests
