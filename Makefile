.PHONY: sync seed serve examples test graph cube-check check-secrets

sync:
	uv sync --extra seed

seed:
	uv run --extra seed python warehouse/seed.py

serve:
	uv run uvicorn app.web.api:app --host 127.0.0.1 --port 8000

examples:
	uv run python examples.py

test:
	uv run pytest -q

graph:
	uv run python -c "from app.agent.graph import mermaid; open('docs/graph.md', 'w').write('\`\`\`mermaid\n' + mermaid() + '\n\`\`\`\n')"

# Pre-warm Cube Cloud (readyz + signed /meta) before recording or running examples.
cube-check:
	uv run python -c "from app.config import settings; from app.models.catalog import VIEW; from app.tools.cube import CubeClient; import httpx; c = CubeClient(settings.cube_url, settings.cube_api_secret); print('readyz:', httpx.get(settings.cube_url + '/readyz', timeout=30).status_code); names = [x['name'] for x in c.meta()['cubes']]; print('meta: 200 · view', VIEW, 'found' if VIEW in names else 'MISSING', names)"

# Fails (exit 1) if anything that looks like a key or a cloud hostname is in the tree.
check-secrets:
	@if grep -rnE 'sk-or-v1-[0-9a-f]{8}|sk-lf-[0-9a-f]{8}-|[a-z0-9-]+\.clickhouse\.cloud|[a-z0-9-]+\.cubecloudapp\.dev' \
	   --exclude-dir=.venv --exclude-dir=.git --exclude-dir=__pycache__ --exclude=.env --exclude=.env.example .; then \
	   echo "SECRETS FOUND"; exit 1; else echo "clean"; fi
