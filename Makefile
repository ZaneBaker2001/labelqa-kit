.PHONY: test lint format precommit

test:
	pytest -q

format:
	pre-commit run black --all-files

lint:
	pre-commit run flake8 --all-files

precommit:
	pre-commit install
