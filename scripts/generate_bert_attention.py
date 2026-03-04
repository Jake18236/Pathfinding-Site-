#!/usr/bin/env python3
"""
Generate attention data for the attention visualizer.

Extracts Q/K/V matrices, attention weights, and PCA-projected embeddings
from BERT and GPT-2 for preset sentences. Outputs JSON files to
static/data/attention/.

Usage:
    python scripts/generate_bert_attention.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2Model,
)


SENTENCES = {
    "default": "I like cute kitties and",
    "short": "the cat sat on",
    "long": "hello world !",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "static" / "data" / "attention"


def truncate(val, decimals=4):
    """Round a float or nested list to given decimal places."""
    if isinstance(val, (list, np.ndarray)):
        return [truncate(v, decimals) for v in val]
    return round(float(val), decimals)


# ── BERT extraction ─────────────────────────────────────────────────

def extract_bert_qkv(model, input_ids, attention_mask):
    qkv_data = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = input[0]
            batch_size, seq_len, _ = hidden_states.shape
            num_heads = module.num_attention_heads
            head_dim = module.attention_head_size

            Q = module.query(hidden_states)
            K = module.key(hidden_states)
            V = module.value(hidden_states)

            Q = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

            qkv_data[layer_idx] = {
                'Q': Q.detach().cpu(), 'K': K.detach().cpu(), 'V': V.detach().cpu(),
            }
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.encoder.layer):
        hooks.append(layer.attention.self.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        output_attentions=True, output_hidden_states=True)

    for h in hooks:
        h.remove()

    return outputs, qkv_data


# ── GPT-2 extraction ────────────────────────────────────────────────

def extract_gpt2_qkv(model, input_ids, attention_mask):
    qkv_data = {}

    def make_pre_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            batch_size, seq_len, _ = hidden_states.shape
            num_heads = module.num_heads
            head_dim = module.head_dim

            # GPT-2 c_attn projects to 3*n_embd (Q, K, V concatenated)
            qkv = module.c_attn(hidden_states)
            Q, K, V = qkv.split(module.embed_dim, dim=2)

            Q = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

            qkv_data[layer_idx] = {
                'Q': Q.detach().cpu(), 'K': K.detach().cpu(), 'V': V.detach().cpu(),
            }
        return hook_fn

    hooks = []
    for i, block in enumerate(model.h):
        hooks.append(block.attn.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        output_attentions=True, output_hidden_states=True)

    for h in hooks:
        h.remove()

    return outputs, qkv_data


# ── Shared processing ───────────────────────────────────────────────

def build_layers_data(attentions, qkv_data, num_layers, num_heads):
    layers_data = []
    for layer_idx in range(num_layers):
        attn_weights = attentions[layer_idx][0].detach().cpu().numpy()
        layer_qkv = qkv_data[layer_idx]

        heads_data = []
        for head_idx in range(num_heads):
            Q = layer_qkv['Q'][0, head_idx].numpy()
            K = layer_qkv['K'][0, head_idx].numpy()
            V = layer_qkv['V'][0, head_idx].numpy()
            weights = attn_weights[head_idx]
            output = weights @ V

            heads_data.append({
                "head": head_idx,
                "Q": truncate(Q.tolist()),
                "K": truncate(K.tolist()),
                "V": truncate(V.tolist()),
                "attention_weights": truncate(weights.tolist()),
                "output": truncate(output.tolist()),
            })

        layers_data.append({"layer": layer_idx, "heads": heads_data})
    return layers_data


def pca_embeddings(hidden_states_0, N):
    embeddings_raw = hidden_states_0[0].detach().cpu().numpy()
    pca = PCA(n_components=min(8, N))
    embeddings_pca = pca.fit_transform(embeddings_raw)
    if embeddings_pca.shape[1] < 8:
        pad = np.zeros((N, 8 - embeddings_pca.shape[1]))
        embeddings_pca = np.concatenate([embeddings_pca, pad], axis=1)
    return embeddings_pca


def process_bert(model, tokenizer, sentence, key):
    print(f"  BERT '{key}': \"{sentence}\"")
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"    Tokens ({len(tokens)}): {tokens}")

    outputs, qkv_data = extract_bert_qkv(model, inputs["input_ids"], inputs["attention_mask"])
    N = len(tokens)
    embeddings_pca = pca_embeddings(outputs.hidden_states[0], N)

    return {
        "model": "bert-base-uncased",
        "tokens": tokens,
        "embeddings_pca": truncate(embeddings_pca.tolist()),
        "layers": build_layers_data(
            outputs.attentions, qkv_data,
            len(outputs.attentions), outputs.attentions[0].shape[1]),
    }


def process_gpt2(model, tokenizer, sentence, key):
    print(f"  GPT-2 '{key}': \"{sentence}\"")
    inputs = tokenizer(sentence, return_tensors="pt")
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Clean BPE Ġ prefix (leading space marker) for display
    tokens = [t.replace("Ġ", "") for t in raw_tokens]
    print(f"    Tokens ({len(tokens)}): {tokens}")

    attention_mask = inputs["attention_mask"]
    outputs, qkv_data = extract_gpt2_qkv(model, inputs["input_ids"], attention_mask)
    N = len(tokens)
    embeddings_pca = pca_embeddings(outputs.hidden_states[0], N)

    return {
        "model": "gpt2",
        "tokens": tokens,
        "embeddings_pca": truncate(embeddings_pca.tolist()),
        "layers": build_layers_data(
            outputs.attentions, qkv_data,
            len(outputs.attentions), outputs.attentions[0].shape[1]),
    }


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"    Wrote {path.name} ({size_mb:.2f} MB)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── BERT ──
    print("Loading bert-base-uncased...")
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    for key, sentence in SENTENCES.items():
        data = process_bert(bert_model, bert_tok, sentence, key)
        write_json(data, OUTPUT_DIR / f"bert-{key}.json")

    del bert_model, bert_tok

    # ── GPT-2 ──
    print("Loading gpt2...")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
    gpt2_model.eval()

    for key, sentence in SENTENCES.items():
        data = process_gpt2(gpt2_model, gpt2_tok, sentence, key)
        write_json(data, OUTPUT_DIR / f"gpt2-{key}.json")

    print("Done!")


if __name__ == "__main__":
    main()
