#!/usr/bin/env python3
"""
Generate attention data for the transformer visualizer.

Extracts Q/K/V matrices, attention weights, and PCA-projected embeddings
from BERT, GPT-2, and MarianMT (encoder-decoder) for preset sentences.
Outputs JSON files to static/data/attention/.

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
    MarianTokenizer, MarianMTModel,
)


# Sentences matching the viz's 3 examples
SENTENCES = [
    "The cat sat on the mat",
    "I like cute kitties and",
    "King is to queen as man",
]

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


# ── MarianMT extraction ─────────────────────────────────────────────

def extract_marian_attention(model, input_ids, attention_mask, decoder_input_ids):
    """Extract attention data from MarianMT using hidden states + weight matrices.

    Avoids hooks since Marian passes args as kwargs which breaks standard hooks.
    Instead, runs forward pass to get attention weights and hidden states,
    then manually computes Q/K/V from the weight matrices.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            output_hidden_states=True,
        )

    num_heads = model.config.encoder_attention_heads
    head_dim = model.config.d_model // num_heads
    num_enc_layers = len(model.model.encoder.layers)
    num_dec_layers = len(model.model.decoder.layers)

    # Compute encoder Q/K/V from hidden states
    enc_qkv = {}
    for i, layer in enumerate(model.model.encoder.layers):
        hs = outputs.encoder_hidden_states[i]  # hidden state input to this layer
        attn = layer.self_attn
        bs, seq_len, _ = hs.shape
        Q = attn.q_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        K = attn.k_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        V = attn.v_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        enc_qkv[i] = {'Q': Q.detach().cpu(), 'K': K.detach().cpu(), 'V': V.detach().cpu()}

    # Compute decoder self-attn Q/K/V and cross-attn Q from hidden states
    dec_qkv = {}
    cross_qkv = {}
    for i, layer in enumerate(model.model.decoder.layers):
        hs = outputs.decoder_hidden_states[i]  # hidden state input to this decoder layer
        bs, seq_len, _ = hs.shape

        # Self-attention Q/K/V
        self_attn = layer.self_attn
        Q = self_attn.q_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        K = self_attn.k_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        V = self_attn.v_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        dec_qkv[i] = {'Q': Q.detach().cpu(), 'K': K.detach().cpu(), 'V': V.detach().cpu()}

        # Cross-attention: Q from decoder after self-attn (approximate with input hs)
        cross_attn = layer.encoder_attn
        cQ = cross_attn.q_proj(hs).view(bs, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        cross_qkv[i] = {'Q': cQ.detach().cpu()}

    return outputs, enc_qkv, dec_qkv, cross_qkv


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


def process_bert(model, tokenizer, sentence, idx):
    print(f"  BERT {idx}: \"{sentence}\"")
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"    Tokens ({len(tokens)}): {tokens}")

    outputs, qkv_data = extract_bert_qkv(model, inputs["input_ids"], inputs["attention_mask"])
    N = len(tokens)
    num_heads = outputs.attentions[0].shape[1]
    num_layers = len(outputs.attentions)
    head_dim = model.config.hidden_size // num_heads
    embeddings_pca = pca_embeddings(outputs.hidden_states[0], N)

    return {
        "model": "bert-base-uncased",
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_size": model.config.hidden_size,
        "head_dim": head_dim,
        "tokens": tokens,
        "embeddings_pca": truncate(embeddings_pca.tolist()),
        "layers": build_layers_data(
            outputs.attentions, qkv_data, num_layers, num_heads),
    }


def process_gpt2(model, tokenizer, sentence, idx):
    print(f"  GPT-2 {idx}: \"{sentence}\"")
    inputs = tokenizer(sentence, return_tensors="pt")
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Clean BPE Ġ prefix (leading space marker) for display
    tokens = [t.replace("Ġ", "") for t in raw_tokens]
    print(f"    Tokens ({len(tokens)}): {tokens}")

    attention_mask = inputs["attention_mask"]
    outputs, qkv_data = extract_gpt2_qkv(model, inputs["input_ids"], attention_mask)
    N = len(tokens)
    num_heads = outputs.attentions[0].shape[1]
    num_layers = len(outputs.attentions)
    head_dim = model.config.n_embd // num_heads
    embeddings_pca = pca_embeddings(outputs.hidden_states[0], N)

    return {
        "model": "gpt2",
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_size": model.config.n_embd,
        "head_dim": head_dim,
        "tokens": tokens,
        "embeddings_pca": truncate(embeddings_pca.tolist()),
        "layers": build_layers_data(
            outputs.attentions, qkv_data, num_layers, num_heads),
    }


def process_marian(model, tokenizer, sentence, idx):
    print(f"  MarianMT {idx}: \"{sentence}\"")

    # Encode source
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    source_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Clean SentencePiece ▁ prefix for display
    source_tokens_clean = [t.replace("▁", " ").strip() or t for t in source_tokens]
    print(f"    Source tokens ({len(source_tokens)}): {source_tokens}")

    # Generate target to get decoder input IDs
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=30,
            num_beams=1,
        )
    target_tokens = tokenizer.convert_ids_to_tokens(generated[0])
    target_tokens_clean = [t.replace("▁", " ").strip() or t for t in target_tokens]
    print(f"    Target tokens ({len(target_tokens)}): {target_tokens}")

    # Run full forward pass with generated decoder input
    outputs, enc_qkv, dec_qkv, cross_qkv = extract_marian_attention(
        model, inputs["input_ids"], inputs["attention_mask"], generated
    )

    num_enc_layers = len(model.model.encoder.layers)
    num_dec_layers = len(model.model.decoder.layers)
    num_heads = model.config.encoder_attention_heads
    head_dim = model.config.d_model // num_heads

    # Build encoder layers data
    encoder_layers = build_layers_data(
        outputs.encoder_attentions, enc_qkv, num_enc_layers, num_heads
    )

    # Build decoder self-attention layers data
    decoder_layers = build_layers_data(
        outputs.decoder_attentions, dec_qkv, num_dec_layers, num_heads
    )

    # Build cross-attention layers data and attach to decoder layers
    for layer_idx in range(num_dec_layers):
        cross_weights = outputs.cross_attentions[layer_idx][0].detach().cpu().numpy()
        cross_q = cross_qkv.get(layer_idx, {})

        heads_data = []
        for head_idx in range(num_heads):
            weights = cross_weights[head_idx]
            Q_data = cross_q.get('Q', None)
            Q = Q_data[0, head_idx].numpy() if Q_data is not None and Q_data.dim() == 4 else []

            heads_data.append({
                "head": head_idx,
                "Q": truncate(Q.tolist()) if hasattr(Q, 'tolist') else [],
                "attention_weights": truncate(weights.tolist()),
            })

        decoder_layers[layer_idx]["cross_attention_heads"] = heads_data

    return {
        "model": "Helsinki-NLP/opus-mt-en-fr",
        "num_heads": num_heads,
        "num_enc_layers": num_enc_layers,
        "num_dec_layers": num_dec_layers,
        "hidden_size": model.config.d_model,
        "head_dim": head_dim,
        "source_tokens": source_tokens_clean,
        "target_tokens": target_tokens_clean,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
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

    for idx, sentence in enumerate(SENTENCES):
        data = process_bert(bert_model, bert_tok, sentence, idx)
        write_json(data, OUTPUT_DIR / f"bert-{idx}.json")

    del bert_model, bert_tok

    # ── GPT-2 ──
    print("Loading gpt2...")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
    gpt2_model.eval()

    for idx, sentence in enumerate(SENTENCES):
        data = process_gpt2(gpt2_model, gpt2_tok, sentence, idx)
        write_json(data, OUTPUT_DIR / f"gpt2-{idx}.json")

    del gpt2_model, gpt2_tok

    # ── MarianMT (en→fr) ──
    print("Loading Helsinki-NLP/opus-mt-en-fr...")
    marian_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr", attn_implementation="eager")
    marian_model.eval()

    for idx, sentence in enumerate(SENTENCES):
        data = process_marian(marian_model, marian_tok, sentence, idx)
        write_json(data, OUTPUT_DIR / f"marian-{idx}.json")

    print("Done!")


if __name__ == "__main__":
    main()
