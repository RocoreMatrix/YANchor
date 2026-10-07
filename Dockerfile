FROM nvcr.io/nvidia/pytorch:26.03-py3
WORKDIR /opt/yanchor-runtime
COPY requirements.txt .
COPY wheels/ ./wheels/
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt \
    && rm -rf wheels requirements.txt
WORKDIR /model
ENTRYPOINT ["python", "-I", "/model/run.py"]
