#!/usr/bin/env python3
"""
Generate training prediction data for Cross-Entropy Loss visualization using text/next-token prediction.

This script trains a tiny transformer on simple text patterns with a 10-token vocabulary
to demonstrate how cross-entropy loss applies to language modeling.

Uses SOFT LABELS based on empirical token distribution from the corpus, so contexts like
"the" can have multiple valid next tokens (cat 40%, dog 40%, etc.) instead of a single
one-hot target.

Output: static/json/cross-entropy-text.json
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Configuration
NUM_SAMPLES = 10  # Number of context samples to track
SNAPSHOT_EPOCHS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25]
TOTAL_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Minimum entropy threshold for "interesting" samples
MIN_ENTROPY = 0.5  # Only select samples with meaningful ambiguity

# 10-token vocabulary (matches existing 10-class setup)
VOCAB = ['[PAD]', 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'a', '.']
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCAB)}
ID_TO_TOKEN = {i: token for i, token in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# Training sentences - carefully designed to create interesting soft label distributions
# Each context should have multiple plausible continuations
SENTENCES = [
    # "the" -> cat or dog (50/50)
    "the cat sat on the mat .",
    "the cat sat on a mat .",
    "the cat ran .",
    "the dog sat on the mat .",
    "the dog sat on a mat .",
    "the dog ran .",

    # "a" -> cat or dog (50/50)
    "a cat sat on the mat .",
    "a cat sat on a mat .",
    "a cat ran .",
    "a dog sat on the mat .",
    "a dog sat on a mat .",
    "a dog ran .",

    # "the/a cat" -> sat or ran (varied)
    "the cat sat .",
    "the cat ran .",
    "a cat sat .",
    "a cat ran .",

    # "the/a dog" -> sat or ran (varied)
    "the dog sat .",
    "the dog ran .",
    "a dog sat .",
    "a dog ran .",

    # "sat on" -> the or a (50/50)
    # Already covered above, but add more for balance
    "the cat sat on the mat .",
    "the cat sat on a mat .",
    "the dog sat on the mat .",
    "the dog sat on a mat .",
    "a cat sat on the mat .",
    "a cat sat on a mat .",
    "a dog sat on the mat .",
    "a dog sat on a mat .",

    # "sat" alone -> on or . (creates ambiguity)
    "the cat sat .",
    "the dog sat .",
    "a cat sat .",
    "a dog sat .",
]

# Output path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'static', 'json', 'cross-entropy-text.json')


def tokenize(sentence):
    """Convert sentence to list of token IDs."""
    tokens = sentence.lower().split()
    return [TOKEN_TO_ID.get(t, TOKEN_TO_ID['[PAD]']) for t in tokens]


def build_context_distributions():
    """
    Build empirical next-token distributions for each context.
    Returns dict: context_tuple -> {next_token_id: count}
    """
    context_counts = defaultdict(lambda: defaultdict(int))

    for sentence in SENTENCES:
        token_ids = tokenize(sentence)
        for i in range(len(token_ids) - 1):
            context = tuple(token_ids[:i+1])
            next_token = token_ids[i+1]
            context_counts[context][next_token] += 1

    return context_counts


def counts_to_distribution(counts):
    """Convert count dict to probability distribution array."""
    total = sum(counts.values())
    dist = [0.0] * VOCAB_SIZE
    for token_id, count in counts.items():
        dist[token_id] = count / total
    return dist


def calculate_entropy(dist):
    """Calculate entropy of a distribution."""
    return -sum(p * math.log(p + 1e-10) for p in dist if p > 0)


def create_training_data():
    """Create context-target pairs from sentences with soft labels."""
    context_counts = build_context_distributions()

    contexts = []
    targets = []  # Still single targets for training
    soft_labels = []  # Soft distributions for evaluation

    for sentence in SENTENCES:
        token_ids = tokenize(sentence)
        for i in range(len(token_ids) - 1):
            context = token_ids[:i+1]
            context_tuple = tuple(context)
            target = token_ids[i+1]

            contexts.append(context)
            targets.append(target)
            soft_labels.append(counts_to_distribution(context_counts[context_tuple]))

    return contexts, targets, soft_labels, context_counts


def pad_sequence(seq, max_len, pad_id=0):
    """Pad sequence to max_len."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TinyTransformer(nn.Module):
    """
    Tiny transformer for next-token prediction.
    2 layers, small embedding dimension, designed for learning simple patterns.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, nhead=2, num_layers=2, max_len=16):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        # x: (batch, seq_len) - token IDs
        # Create causal mask
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Transform
        x = self.transformer(x, mask=mask, is_causal=True)

        # Output logits for last position
        logits = self.fc_out(x[:, -1, :])
        return logits

    def predict_proba(self, x):
        """Get softmax probabilities for next token."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def select_ambiguous_samples(context_counts, num_samples=10, min_entropy=MIN_ENTROPY):
    """
    Select samples with interesting ambiguities (high entropy distributions).
    Only includes contexts where multiple next tokens are plausible.
    """
    samples = []

    # Score all contexts by entropy
    scored_contexts = []
    for context_tuple, counts in context_counts.items():
        dist = counts_to_distribution(counts)
        entropy = calculate_entropy(dist)

        # Only consider contexts with meaningful ambiguity
        if entropy >= min_entropy:
            # Count how many tokens have significant probability
            num_options = sum(1 for p in dist if p >= 0.1)
            scored_contexts.append({
                'context_tuple': context_tuple,
                'counts': counts,
                'dist': dist,
                'entropy': entropy,
                'num_options': num_options
            })

    # Sort by: entropy (descending), then number of options (descending), then context length (ascending)
    scored_contexts.sort(key=lambda x: (-x['entropy'], -x['num_options'], len(x['context_tuple'])))

    print(f"\nFound {len(scored_contexts)} contexts with entropy >= {min_entropy}:")
    for ctx in scored_contexts[:15]:  # Show top 15
        context_str = ' '.join(ID_TO_TOKEN[t] for t in ctx['context_tuple'])
        dist_str = format_distribution(ctx['dist'])
        print(f"  '{context_str}' -> [{dist_str}] (H={ctx['entropy']:.2f})")

    # Select diverse samples
    selected_contexts = set()
    for ctx in scored_contexts:
        if len(samples) >= num_samples:
            break

        context_tuple = ctx['context_tuple']

        # Skip if we already have this exact context
        if context_tuple in selected_contexts:
            continue

        # Get the most likely next token (for display purposes)
        argmax_token = max(ctx['counts'].keys(), key=lambda k: ctx['counts'][k])
        context_str = ' '.join(ID_TO_TOKEN[t] for t in context_tuple)

        samples.append({
            'context_ids': list(context_tuple),
            'target_id': argmax_token,
            'context_str': context_str,
            'soft_distribution': ctx['dist'],
            'entropy': ctx['entropy']
        })
        selected_contexts.add(context_tuple)

    return samples


def get_predictions(model, samples, device, max_len):
    """Get model predictions for all sample contexts."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for sample in samples:
            ctx = pad_sequence(sample['context_ids'], max_len)
            ctx_tensor = torch.tensor([ctx], dtype=torch.long).to(device)
            probs = model.predict_proba(ctx_tensor)
            predictions.append(probs.cpu().numpy()[0])

    return predictions


def calculate_cross_entropy(expected_dist, predicted_dist, epsilon=1e-15):
    """Calculate cross-entropy loss between two distributions."""
    loss = 0.0
    for p_true, p_pred in zip(expected_dist, predicted_dist):
        if p_true > 0:
            loss -= p_true * math.log(max(p_pred, epsilon))
    return loss


def train_epoch(model, contexts, targets, optimizer, device, max_len):
    """Train for one epoch using hard labels (standard practice)."""
    model.train()

    # Shuffle data
    indices = torch.randperm(len(contexts))
    total_loss = 0
    correct = 0

    for i in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[i:i+BATCH_SIZE]

        # Prepare batch
        batch_ctx = []
        batch_tgt = []
        for idx in batch_indices:
            ctx = pad_sequence(contexts[idx], max_len)
            batch_ctx.append(ctx)
            batch_tgt.append(targets[idx])

        batch_ctx = torch.tensor(batch_ctx, dtype=torch.long).to(device)
        batch_tgt = torch.tensor(batch_tgt, dtype=torch.long).to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_ctx)
        loss = F.cross_entropy(logits, batch_tgt)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_indices)
        pred = logits.argmax(dim=1)
        correct += (pred == batch_tgt).sum().item()

    return total_loss / len(contexts), correct / len(contexts)


def format_distribution(dist):
    """Format distribution for display, showing non-zero probabilities."""
    parts = []
    for i, p in enumerate(dist):
        if p > 0.01:  # Only show significant probabilities
            parts.append(f"{VOCAB[i]}:{p:.0%}")
    return ", ".join(parts)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Vocabulary: {VOCAB}")
    print(f"Training tiny transformer for next-token prediction with SOFT LABELS")
    print(f"Minimum entropy threshold: {MIN_ENTROPY}")
    print()

    # Create training data with soft labels
    contexts, targets, soft_labels, context_counts = create_training_data()
    print(f"Created {len(contexts)} training examples from {len(SENTENCES)} sentences")

    # Find max context length
    max_len = max(len(ctx) for ctx in contexts)
    print(f"Max context length: {max_len}")

    # Select only ambiguous samples (high entropy)
    samples = select_ambiguous_samples(context_counts, NUM_SAMPLES, MIN_ENTROPY)

    if len(samples) < NUM_SAMPLES:
        print(f"\nWarning: Only found {len(samples)} samples with entropy >= {MIN_ENTROPY}")

    print(f"\n=== Selected {len(samples)} samples for tracking ===")
    for i, s in enumerate(samples):
        dist_str = format_distribution(s['soft_distribution'])
        print(f"  {i}: '{s['context_str']}' -> [{dist_str}] (H={s['entropy']:.2f})")

    # Prepare sample data for output
    sample_data = []
    for i, sample in enumerate(samples):
        # Find the dominant class (argmax of soft distribution)
        dominant_class = sample['soft_distribution'].index(max(sample['soft_distribution']))

        sample_data.append({
            'index': i,
            'trueLabel': dominant_class,  # For display - most likely token
            'expectedDistribution': sample['soft_distribution'],  # Soft labels!
            'context': sample['context_str'],
            'contextTokens': sample['context_ids'],
            'snapshots': []
        })

    # Initialize model
    model = TinyTransformer(max_len=max_len + 1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop with snapshot capture
    print(f"\nTraining for {TOTAL_EPOCHS} epochs...")

    for epoch in range(TOTAL_EPOCHS + 1):
        # Capture snapshot at specified epochs
        if epoch in SNAPSHOT_EPOCHS:
            print(f"  Capturing snapshot at epoch {epoch}...")
            predictions = get_predictions(model, samples, DEVICE, max_len)

            for i, (sample, preds) in enumerate(zip(samples, predictions)):
                soft_dist = sample['soft_distribution']
                loss = calculate_cross_entropy(soft_dist, preds)
                predicted_class = int(preds.argmax())
                dominant_class = soft_dist.index(max(soft_dist))

                snapshot = {
                    'epoch': epoch,
                    'predictions': [float(p) for p in preds],
                    'loss': float(loss),
                    'predictedClass': predicted_class,
                    'correct': predicted_class == dominant_class
                }
                sample_data[i]['snapshots'].append(snapshot)

        # Train one epoch (skip epoch 0 - that's before training)
        if epoch < TOTAL_EPOCHS:
            train_loss, train_acc = train_epoch(model, contexts, targets, optimizer, DEVICE, max_len)

            if epoch % 5 == 0 or epoch == TOTAL_EPOCHS - 1:
                print(f"  Epoch {epoch + 1}/{TOTAL_EPOCHS}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Calculate final cross-entropy against soft labels
    final_predictions = get_predictions(model, samples, DEVICE, max_len)
    final_losses = [calculate_cross_entropy(s['soft_distribution'], p)
                    for s, p in zip(samples, final_predictions)]
    avg_final_loss = sum(final_losses) / len(final_losses)

    # Also calculate accuracy against dominant class
    final_correct = sum(1 for s, p in zip(samples, final_predictions)
                       if p.argmax() == s['soft_distribution'].index(max(s['soft_distribution'])))
    final_acc = final_correct / len(samples)

    print(f"\nFinal average cross-entropy (soft labels): {avg_final_loss:.3f}")
    print(f"Final accuracy (vs dominant class): {final_acc * 100:.1f}%")

    # Prepare output JSON
    output = {
        'metadata': {
            'dataset': 'text',
            'numSamples': len(samples),
            'numClasses': VOCAB_SIZE,
            'totalEpochs': TOTAL_EPOCHS,
            'snapshotEpochs': SNAPSHOT_EPOCHS,
            'numSnapshots': len(SNAPSHOT_EPOCHS),
            'finalAvgLoss': round(avg_final_loss, 3),
            'generated': datetime.now().isoformat(),
            'classLabels': VOCAB,
            'softLabels': True  # Flag indicating this dataset uses soft labels
        },
        'samples': sample_data
    }

    # Write to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {OUTPUT_PATH}")

    # Print summary
    print("\nSample summary:")
    for sample in sample_data:
        first_loss = sample['snapshots'][0]['loss']
        last_loss = sample['snapshots'][-1]['loss']
        last_pred = sample['snapshots'][-1]['predictedClass']
        dominant = sample['trueLabel']
        context = sample['context']
        dist_str = format_distribution(sample['expectedDistribution'])
        print(f"  '{context}' -> [{dist_str}]")
        print(f"    Loss {first_loss:.3f} -> {last_loss:.3f}, "
              f"Final: '{VOCAB[last_pred]}' "
              f"({'dominant' if last_pred == dominant else 'not dominant'})")


if __name__ == '__main__':
    main()
