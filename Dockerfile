FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=4
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# bake the BERT-NSP judge and base distilGPT-2 into the image (no runtime downloads)
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM, BertForNextSentencePrediction, BertTokenizer; \
BertTokenizer.from_pretrained('bert-base-uncased'); BertForNextSentencePrediction.from_pretrained('bert-base-uncased'); \
AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2')"
COPY . .
EXPOSE 8080
ENV PORT=8080
CMD ["python", "app.py"]
