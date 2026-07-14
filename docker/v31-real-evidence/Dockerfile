FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE /workspace/

# This arm64 image is a development executor. Release evidence continues to use
# the separately hashed Linux CPU profile; the runtime manifest records that
# distinction and the claim gate remains closed for this image.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       'torch==2.11.0' 'torchvision==0.26.0' \
    && python -m pip install --no-cache-dir \
       'numpy==2.2.6' 'Pillow>=10.0' 'PyYAML>=6.0' \
       'transformers==5.5.4' 'accelerate==1.14.0' 'num2words==0.5.14' \
    && python -m pip check \
    && python -c "import importlib.util as i, torch; assert not torch.cuda.is_available(); assert i.find_spec('triton') is None"

COPY . /workspace
RUN python -m pip install --no-cache-dir --no-deps .

ENTRYPOINT ["lunavla-v3"]
