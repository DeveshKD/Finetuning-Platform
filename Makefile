.PHONY: api scheduler worker test

api:
	bash scripts\start_api.sh

scheduler:
	bash scripts\run_scheduler.sh

worker:
	bash scripts\run_worker.sh $(JOB_ID)

test:
	pytest -q
