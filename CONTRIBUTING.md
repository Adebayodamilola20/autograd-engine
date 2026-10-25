# Contributing to nabla

Thanks for helping. nabla is a teaching-grade autodiff engine, so changes are
judged on two things: **is it correct** (gradient-checked, tested) and **is it
clear** (someone learning from the code can follow it).

## Set up

```bash
git clone https://github.com/Adebayodamilola20/autograd-engine.git && cd autograd-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install -e . --no-deps
python -m pytest            # everything should pass before you start
```

Or open the repo in a devcontainer / build the `Dockerfile`; both use the same
pinned `requirements.lock`.

## Making a change

1. **One change per commit, with its tests in the same commit.** A new op, a
   bug fix or a refactor each get their own commit. Don't mix formatting with
   behaviour changes.
2. **New differentiable operations need a gradient check.** Add a case to
   `tests/test_gradients.py` comparing `backward()` against finite differences
   (`nabla.core.gradcheck`). This is the project's core guarantee.
3. **Keep coverage up.** CI runs `pytest --cov=nabla` and fails under 80%.
4. **Lint:** `ruff check .` must pass.
5. **Dependencies:** change them in `pyproject.toml`, then regenerate the lock:

   ```bash
   uv pip compile pyproject.toml --extra dev --extra viz --universal --python-version 3.10 -o requirements.lock
   ```

   CI fails if `requirements.lock` is out of date.
6. **Results:** if a change affects a headline number, rerun it from its config
   (`--config configs/...`) and update the README and `runs/reference/`.

## Pull requests

Describe *why* as well as what, link any issue, and note anything you measured.
All CI jobs (tests on 3.10–3.13, coverage gate, lint, lockfile, audit, Docker
build) must be green.

## Reporting bugs

Open an issue with the smallest snippet that shows the problem, what you
expected, and what happened (full traceback if there is one).
