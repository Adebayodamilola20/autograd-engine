# Reproducible environment: the pinned lockfile on a fixed Python.
#   docker build -t nabla .
#   docker run --rm nabla                                   # runs the test suite
#   docker run --rm nabla python examples/train_xor.py      # or anything else
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# dependencies first, so code changes don't re-install them
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
RUN pip install --no-cache-dir -e . --no-deps

CMD ["python", "-m", "pytest"]
