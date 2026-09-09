.PHONY: seed smoke golden eval-trivial eval-simple eval-pipeline

seed:
	python3 data/generate_seed_data.py

smoke: seed
	python3 -m src.ingest
	python3 -m src.pipeline

golden:
	python3 -m eval.build_golden_set --n 200

eval-trivial:
	python3 -m eval.harness --system trivial

eval-simple:
	python3 -m eval.harness --system simple

eval-pipeline:
	python3 -m eval.harness --system pipeline --judge --judge_sample_n 30
