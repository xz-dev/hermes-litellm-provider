.PHONY: check deploy restart test

REMOTE ?= xz@100.65.231.22
PLUGIN_DIR ?= ~/.hermes/plugins/model-providers/litellm
MODEL ?= glm-5.1

check:
	python -m py_compile __init__.py

deploy: check
	ssh $(REMOTE) 'mkdir -p $(PLUGIN_DIR)'
	scp __init__.py plugin.yaml $(REMOTE):$(PLUGIN_DIR)/
	ssh $(REMOTE) 'rm -rf $(PLUGIN_DIR)/__pycache__'

restart:
	ssh $(REMOTE) 'systemctl --user restart hermes-gateway.service && systemctl --user is-active hermes-gateway.service litellm.service hermes-webui.service'

test:
	ssh $(REMOTE) 'set -a; source ~/.config/litellm/litellm.env 2>/dev/null || true; set +a; hermes --provider litellm -m $(MODEL) -z "Reply only: OK"'
