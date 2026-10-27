# Changelog

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `mypy` type-check job in CI; the package is type-clean (42 errors fixed).
- `RMSprop` optimiser, and learning-rate schedules (`nabla.optim.schedules`: constant, step decay, exponential, cosine, warmup) for `Trainer.fit(lr_schedule=...)`.
- Offline tests for the MNIST loader (IDX parsing, caching, subsets).

### Fixed
- `Dataset` accepts image-shaped samples, so `load_mnist(flatten=False)` works.

## [0.2.1] - 2026-10-26

### Changed
- Dependencies are locked with `uv.lock` (the standard uv lockfile); CI installs with `uv sync --locked`, and `requirements.txt` is now the fully pinned export of it. Replaces `requirements.lock`.

### Added
- CI `experiment` job: trains MNIST from `configs/mnist_ci.json`, writes the run record, fails below 94% test accuracy, and uploads the record as an artifact.
- `python -m nabla.training.runs check` to hold a run record to a threshold.

## [0.2.0] - 2026-10-25

### Added
- `requirements.lock`: every dependency pinned for Python 3.10–3.13 on all platforms; CI installs from it and fails if it drifts from `pyproject.toml`.
- 80% coverage gate in CI (suite at ~88%).
- `pip-audit` dependency audit job and Dependabot updates for pip and GitHub Actions.
- Tests for the `Trainer` (17% → 97% coverage) and every plotting function (0% → 98%).
- `nabla.training.runs`: JSON experiment configs (`configs/`) and a run record for every training run (`runs/`); `configs/mnist.json` reproduces the 97.63% result, recorded in `runs/reference/mnist.json`.
- `--config` for `examples/train_mnist.py` and `examples/train_language_model.py`.
- `CONTRIBUTING.md`, this changelog, a `Dockerfile` and a devcontainer.

## [0.1.0] - 2026-09-19

### Added
- Scalar (`Value`) and tensor (`Tensor`) reverse-mode autodiff engines with finite-difference gradient checking.
- MLPs on both engines, SGD/Adam/AdamW, losses, data loading and the `Trainer`.
- MNIST training to 97.63% test accuracy; a GPT-style transformer trained on tinyshakespeare; spiking and energy-based networks.
- `nabla` CLI, graph renderers, plotting, web demo, benchmarks against PyTorch, experiment sweeps and 21 chapters of documentation.

[Unreleased]: https://github.com/Adebayodamilola20/autograd-engine/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Adebayodamilola20/autograd-engine/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Adebayodamilola20/autograd-engine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Adebayodamilola20/autograd-engine/releases/tag/v0.1.0
