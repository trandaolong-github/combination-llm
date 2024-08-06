VIRTUALENV ?= $(CURDIR)/env
PYTHONVERSION ?= python3
PYTHON = $(VIRTUALENV)/bin/$(notdir $(PYTHONVERSION))

.PHONY: venv
venv: $(VIRTUALENV)

$(VIRTUALENV): $(VIRTUALENV)/freeze.txt
$(VIRTUALENV)/freeze.txt: requirements.txt
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --requirement=$<

.PHONY: clean
clean:
	# Only remove files and directories ignored by Git. This should clean all
	# temp files, while simultaneously not delete those yet to be committed.
	git clean -Xdf
