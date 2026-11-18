import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ART = Path(__file__).parent / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# 1. Workshop custom BPE tokenizer
# ---------------------------------------------------------------------
tok = PreTrainedTokenizerFast(
    tokenizer_file=str(ART / "tokenizer" / "tokenizer.json"),
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    eos_token="<|endoftext|>",
    bos_token="<|endoftext|>",
)

# ---------------------------------------------------------------------
# 2. MiniGPT-2 architecture from the workshop notebook
# ---------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out, self.num_heads = d_out, num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, n, _ = x.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.W_query(x).view(shape).transpose(1, 2)
        k = self.W_key(x).view(shape).transpose(1, 2)
        v = self.W_value(x).view(shape).transpose(1, 2)
        scores = q @ k.transpose(2, 3)
        scores = scores.masked_fill(self.mask.bool()[:n, :n], -torch.inf)
        weights = self.dropout(torch.softmax(scores / self.head_dim ** 0.5, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, n, self.d_out)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                      cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        return x + self.ff(self.norm2(x))


class MiniGPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm_f = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        _, n = idx.shape
        pos = torch.arange(n, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        return self.head(self.norm_f(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg["context_length"]:])[:, -1, :]
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


mini_cfg = {
    "vocab_size": tok.vocab_size,
    "context_length": 128,
    "emb_dim": 192,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
mini_model = MiniGPT2(mini_cfg)
mini_state = torch.load(ART / "mini_gpt2.pt", map_location="cpu", weights_only=True)
mini_model.load_state_dict(mini_state)
mini_model.to(DEVICE).eval()


def generate_mini(prompt, max_new_tokens, temperature):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        output = mini_model.generate(ids, int(max_new_tokens), float(temperature), 40)[0]
    return tok.decode(output.tolist())

# ---------------------------------------------------------------------
# 3. DistilGPT-2 + workshop LoRA adapter
# ---------------------------------------------------------------------
hf_tok = AutoTokenizer.from_pretrained("distilgpt2")
hf_tok.pad_token = hf_tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
lora_model = PeftModel.from_pretrained(base_model, str(ART / "distilgpt2-lora"))
lora_model.to(DEVICE).eval()


def format_prompt(instruction):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    )


def generate_lora(prompt, max_new_tokens, use_instruction_template=True):
    full_prompt = format_prompt(prompt) if use_instruction_template else prompt
    inputs = hf_tok(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = lora_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=hf_tok.eos_token_id,
        )
    decoded = hf_tok.decode(output[0], skip_special_tokens=True)
    return decoded[len(full_prompt):].strip()

# ---------------------------------------------------------------------
# 4. Workshop ChromaDB RAG index
# ---------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = Chroma(
    collection_name="byol",
    persist_directory=str(ART / "chroma"),
    embedding_function=embeddings,
)

RAG_TEMPLATE = """Answer the question using only the context below.
If the context does not contain the answer, say you do not know.

### Context:
{context}

### Question:
{question}

### Answer:
"""


def tokenize_tab(text):
    ids = tok.encode(text)
    rows = [[i, token.replace("\u0120", "_"), token_id]
            for i, (token, token_id) in enumerate(zip(tok.convert_ids_to_tokens(ids), ids))]
    return f"{len(text)} characters → {len(ids)} tokens", rows


def generate_tab(prompt, model_choice, max_tokens, temperature):
    if model_choice.startswith("MiniGPT2"):
        return generate_mini(prompt, max_tokens, temperature)
    return generate_lora(prompt, max_tokens)


def rag_tab(question, k):
    docs = vector_store.similarity_search(question, k=int(k))
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    answer = generate_lora(RAG_TEMPLATE.format(context=context, question=question), 120, use_instruction_template=False)
    sources = "\n\n".join(
        f"**{i}. {Path(doc.metadata.get('source', 'document')).name}**\n\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )
    return answer, sources or "No matching documents were retrieved."


with gr.Blocks(title="Build Your Own LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Build Your Own LLM\nExplore the tokenizer, text-generation models and retrieval-augmented question answering built during the workshop.")

    with gr.Tab("1. Tokenizer"):
        tokenizer_input = gr.Textbox(label="Text", value="LoRA adapters on a laptop.", lines=2)
        tokenizer_button = gr.Button("Tokenize", variant="primary")
        tokenizer_summary = gr.Markdown()
        tokenizer_table = gr.Dataframe(headers=["#", "token", "id"], interactive=False)
        tokenizer_button.click(tokenize_tab, inputs=tokenizer_input,
                               outputs=[tokenizer_summary, tokenizer_table])

    with gr.Tab("2. Generate"):
        prompt_input = gr.Textbox(label="Prompt", value="Rewrite using a simile: the car is very fast.", lines=2)
        model_choice = gr.Radio(
            ["MiniGPT2 (ours)", "DistilGPT2 + LoRA"],
            value="DistilGPT2 + LoRA", label="Model"
        )
        max_tokens = gr.Slider(16, 128, value=64, step=8, label="Max new tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature (MiniGPT2)")
        generate_button = gr.Button("Generate", variant="primary")
        generation_output = gr.Textbox(label="Output", lines=8)
        generate_button.click(generate_tab,
                              inputs=[prompt_input, model_choice, max_tokens, temperature],
                              outputs=generation_output)

    with gr.Tab("3. Ask your documents"):
        question_input = gr.Textbox(label="Question", value="What time does the workshop end?")
        chunks = gr.Slider(1, 6, value=3, step=1, label="Retrieved chunks")
        ask_button = gr.Button("Ask", variant="primary")
        rag_answer = gr.Textbox(label="Answer", lines=6)
        rag_sources = gr.Markdown(label="Retrieved sources")
        ask_button.click(rag_tab, inputs=[question_input, chunks], outputs=[rag_answer, rag_sources])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
