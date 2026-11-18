[README.md](https://github.com/user-attachments/files/35039589/README.md)
# Build Your Own LLM

An interactive Gradio application based on the **Build Your Own LLM: Pretrain, Fine-Tune, and Deploy** workshop. It brings together three learning demonstrations:

1. **Custom tokenizer** – inspect token IDs produced by the workshop's custom BPE tokenizer.
2. **Text generation** – compare a small GPT-style model trained from scratch with DistilGPT-2 adapted using LoRA.
3. **Retrieval-Augmented Generation (RAG)** – retrieve relevant workshop knowledge-base passages from ChromaDB and use them as context for an answer.

## Tech stack

Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, Sentence Transformers, LangChain, ChromaDB, and Gradio.

## Repository structure

```text
.
├── app.py
├── requirements.txt
└── artifacts/
    ├── mini_gpt2.pt
    ├── tokenizer/
    │   └── tokenizer.json
    ├── distilgpt2-lora/
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   └── tokenizer files
    └── chroma/
        └── ChromaDB persistence files
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open the local Gradio URL printed in the terminal (usually `http://127.0.0.1:7860`).

## Notes

- The first launch downloads the base `distilgpt2` model and the `sentence-transformers/all-MiniLM-L6-v2` embedding model from Hugging Face. Internet access is required on first launch.
- The application loads the saved workshop artifacts from the `artifacts/` directory; it does not retrain the models at startup.
- Generated text is a demonstration of the workshop models, not a benchmark of production-grade language-model quality.
- Hosting providers have different memory, CPU, storage, and sleep limits. Confirm that the selected host can run PyTorch, Transformers, and the embedding model before deploying.
