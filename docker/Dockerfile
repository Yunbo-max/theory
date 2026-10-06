FROM python:3.11.13-slim-bookworm
RUN pip install --no-cache-dir evalplus==0.3.1 tree-sitter==0.23.2 tree-sitter-python==0.23.6 transformers==4.51.3 numpy==2.2.6
COPY score.py /opt/score.py
ENV HOME=/tmp XDG_CACHE_HOME=/tmp/cache PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python", "/opt/score.py"]
