# Two-stage image: sim (Isaac-capable) + real (ROS 2).
# Use sim_env/training/replay/adaptation inside the Isaac Lab container.
# Use sim2real/real_world on a ROS 2 Humble host.
#
# For pure unit tests and CI, a minimal Python image is enough.
FROM python:3.10-slim AS testbox

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/go2-phoenix
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN pip install --upgrade pip \
 && pip install ".[dev,real]"

CMD ["pytest", "tests", "-m", "not sim and not ros"]
