PYTHON ?= python3
.PHONY: reproduce validate check
reproduce:
	$(PYTHON) scripts/reproduce.py
validate:
	$(PYTHON) analysis/validate.py
check:
	$(PYTHON) analysis/check_results.py
