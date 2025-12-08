#!/usr/bin/env python3
"""
Generate AORG (Attributed Object Reference Graph) embeddings using pretrained BERT model

This script converts ORG files (JSON graph structures) into PyTorch Geometric Data objects
with BERT-generated node embeddings.

Usage:
    python embed_aorg.py -r <base_dir> -v <vocab_path> -b <bert_model_path> -label <labels_csv>

Example:
    python embed_aorg.py \
        -r data/train \
        -v "../semester project/PDFObj2Vec/models/vocab-65k" \
        -b "../semester project/PDFObj2Vec/models/BERT65k.pth" \
        -label data/train/labels.csv
"""

import csv
import json
import os
import sys
import argparse
import random
import numpy as np
import torch
import tqdm
from pathlib import Path
from torch_geometric.data import Data
from transformers import BertConfig

# Add lib directory to path for custom modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "lib"))

from bert.model import PREBERT
from bert.vocab import WordVocab
from utils.set_random import setup_seed

# Set random seed for reproducibility
setup_seed(42)


class BERTEmbeddingGenerator:
    """Generate BERT embeddings for ORG nodes"""

    def __init__(self, vocab_path, bert_path, hidden_size=512, seq_len=64):
        """
        Initialize BERT model for embedding generation

        Args:
            vocab_path: Path to vocabulary file
            bert_path: Path to pretrained BERT model
            hidden_size: BERT hidden layer size (default: 512)
            seq_len: Maximum sequence length (default: 64)
        """
        self.vocab_path = vocab_path
        self.bert_path = bert_path
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Load vocabulary
        print(f"[BERT] Loading vocabulary from: {vocab_path}")
        self.vocab = WordVocab.load_vocab(vocab_path)
        print(f"[BERT] Vocabulary size: {len(self.vocab)}")

        # Initialize BERT model
        print(f"[BERT] Initializing BERT model (hidden_size={hidden_size})...")
        config = BertConfig(
            hidden_size=self.hidden_size,
            vocab_size=len(self.vocab),
            num_attention_heads=8,
            num_hidden_layers=8,
            return_dict=True,
        )
        self.model = PREBERT(config)

        # Load pretrained weights and setup device
        self.device = self._setup_device()
        self._load_model()

        print(f"[BERT] Model ready on device: {self.device}")

    def _setup_device(self):
        """Determine best available device (CUDA > MPS > CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps:0")
        else:
            return torch.device("cpu")

    def _load_model(self):
        """Load pretrained BERT model weights"""
        print(f"[BERT] Loading pretrained weights from: {self.bert_path}")

        if self.device.type == "mps":
            state_dict = torch.load(self.bert_path, map_location=torch.device("mps"))
        elif self.device.type == "cuda":
            state_dict = torch.load(self.bert_path, map_location=torch.device("cuda"))
        else:
            state_dict = torch.load(self.bert_path, map_location=torch.device("cpu"))

        self.model.load_state_dict(state_dict)
        self.bert = self.model.bert.bert
        self.bert = self.bert.to(self.device)
        self.bert.eval()

    def norm_insn(self, insn_list):
        """
        Normalize instruction list to standardized format

        Args:
            insn_list: List of [key, type, value] instructions

        Returns:
            List of normalized instruction strings
        """
        insn_list_norm = []
        for insn in insn_list:
            if insn[1] == "STREAM":
                insn_norm = insn[1]
            else:
                insn_norm = insn[0] + "_" + insn[1]
            insn_list_norm.append(insn_norm)
        return insn_list_norm

    def generate_embeddings(self, node_instructions):
        """
        Generate BERT embeddings for a list of node instructions

        Args:
            node_instructions: List of instruction lists (one per node)

        Returns:
            Tensor of shape (num_nodes, hidden_size) with node embeddings
        """
        # Prepare tokenized sequences
        tokenized_sequences = []
        segment_labels = []
        attention_masks = []

        for insn_list in node_instructions:
            # Normalize and tokenize
            seq_norm = self.norm_insn(insn_list)
            tokenized_seq = []
            for insn in seq_norm:
                token_id = self.vocab.stoi.get(insn, self.vocab.unk_index)
                tokenized_seq.append(token_id)

            # Add [CLS] and [SEP] tokens
            tokenized_seq = (
                [self.vocab.sos_index] + tokenized_seq + [self.vocab.eos_index]
            )

            # Truncate to max length
            tokenized_seq = tokenized_seq[: self.seq_len]
            segment_label = [0 for _ in range(len(tokenized_seq))][: self.seq_len]
            attention_mask = [1 for _ in range(len(tokenized_seq))][: self.seq_len]

            # Padding
            padding_len = self.seq_len - len(tokenized_seq)
            tokenized_seq.extend([self.vocab.pad_index] * padding_len)
            segment_label.extend([self.vocab.pad_index] * padding_len)
            attention_mask.extend([0] * padding_len)

            tokenized_sequences.append(tokenized_seq)
            segment_labels.append(segment_label)
            attention_masks.append(attention_mask)

        # Convert to tensors
        inputs = torch.tensor(tokenized_sequences, dtype=torch.long)
        segment_labels_tensor = torch.tensor(segment_labels, dtype=torch.long)
        attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long)

        # Generate embeddings in batches
        batch_size = 100
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size].to(self.device)
                batch_segments = segment_labels_tensor[i : i + batch_size].to(
                    self.device
                )
                batch_masks = attention_masks_tensor[i : i + batch_size].to(self.device)

                outputs = self.bert(
                    input_ids=batch_inputs,
                    attention_mask=batch_masks,
                    token_type_ids=batch_segments,
                    return_dict=True,
                )

                embeddings.extend(outputs.pooler_output.cpu().tolist())

        return torch.tensor(embeddings, dtype=torch.float)


def load_labels(labels_csv):
    """Load labels from CSV file"""
    labels = {}
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["Id"]] = int(row["Label"])
    return labels


def process_org_to_aorg(org_path, label, bert_generator):
    """
    Convert ORG file to AORG (Attributed ORG) with BERT embeddings

    Args:
        org_path: Path to ORG JSON file
        label: Graph label (0=benign, 1=malicious)
        bert_generator: BERTEmbeddingGenerator instance

    Returns:
        PyTorch Geometric Data object with node embeddings
    """
    # Load ORG JSON
    with open(org_path, "r", encoding="utf-8") as f:
        org = json.load(f)

    # Build node ID mapping
    addr_to_id = {}
    current_node_id = -1
    node_instructions = []

    for addr, block in org.items():
        current_node_id += 1
        addr_to_id[addr] = current_node_id
        node_instructions.append(block["insn_list"])

    # Generate BERT embeddings for all nodes
    node_embeddings = bert_generator.generate_embeddings(node_instructions)

    # Build edge list
    senkeys = ["/OpenAction", "/Action", "/JavaScript", "/JS", "/S"]
    edge_index = []
    edge_attr = []

    for addr, block in org.items():
        start_nid = addr_to_id[addr]
        insns_flat = [elem for insn in block["insn_list"] for elem in insn]

        for out_edge in block["out_edge_list"]:
            if str(out_edge) in addr_to_id:
                end_nid = addr_to_id[str(out_edge)]
                edge_index.append([start_nid, end_nid])

                # Check if edge involves sensitive keys
                intersection = set(senkeys) & set(insns_flat)
                edge_attr.append(1 if intersection else 0)

    # Create PyTorch Geometric Data object
    x = node_embeddings
    y = torch.tensor([label], dtype=torch.long)
    edge_index_tensor = (
        torch.tensor(edge_index, dtype=torch.long).t()
        if edge_index
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr_tensor = (
        torch.tensor(edge_attr, dtype=torch.float)
        if edge_attr
        else torch.empty(0, dtype=torch.float)
    )

    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y)

    return data


def generate_aorgs(
    base_dir, vocab_path, bert_path, labels_csv, hidden_size=512, seq_len=64
):
    """
    Generate AORGs for all ORG files in dataset

    Args:
        base_dir: Base directory containing org/ subdirectory
        vocab_path: Path to vocabulary file
        bert_path: Path to BERT model
        labels_csv: Path to labels CSV file
        hidden_size: BERT hidden size (default: 512)
        seq_len: Max sequence length (default: 64)
    """
    print("=" * 80)
    print("AORG GENERATION - BERT EMBEDDINGS")
    print("=" * 80)

    # Setup paths
    base_path = Path(base_dir)
    org_dir = base_path / "org"
    output_dir = base_path / "org_after_prebert"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CONFIG]")
    print(f"  Input ORGs:     {org_dir}")
    print(f"  Output AORGs:   {output_dir}")
    print(f"  BERT model:     {bert_path}")
    print(f"  Vocabulary:     {vocab_path}")
    print(f"  Labels:         {labels_csv}")
    print(f"  Hidden size:    {hidden_size}")
    print(f"  Seq length:     {seq_len}")

    # Load labels
    print(f"\n[1/3] Loading labels...")
    labels = load_labels(labels_csv)
    print(f"      Loaded {len(labels)} labels")

    # Initialize BERT
    print(f"\n[2/3] Initializing BERT model...")
    bert_gen = BERTEmbeddingGenerator(vocab_path, bert_path, hidden_size, seq_len)

    # Get ORG files
    org_files = sorted([f for f in org_dir.iterdir() if f.is_file()])
    print(f"\n[3/3] Processing {len(org_files)} ORG files...")

    processed = 0
    skipped = 0
    errors = []

    for idx, org_file in enumerate(tqdm.tqdm(org_files, desc="Generating AORGs")):
        filename = org_file.name
        output_path = output_dir / f"data_{idx}_{filename}.pt"

        # Skip if already processed
        if output_path.exists():
            skipped += 1
            continue

        # Skip if no label
        if filename not in labels:
            errors.append((filename, "No label found"))
            continue

        try:
            # Generate AORG
            aorg_data = process_org_to_aorg(org_file, labels[filename], bert_gen)

            # Save
            torch.save(aorg_data, output_path)
            processed += 1

        except Exception as e:
            errors.append((filename, str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total ORG files:      {len(org_files)}")
    print(f"Processed:            {processed}")
    print(f"Skipped (existing):   {skipped}")
    print(f"Errors:               {len(errors)}")

    if errors:
        print(f"\nErrors encountered:")
        for filename, error in errors[:10]:  # Show first 10
            print(f"  - {filename}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\nAORGs saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate AORG embeddings using BERT")
    parser.add_argument(
        "-r",
        "--base-dir",
        required=True,
        dest="base_dir",
        help="Base directory containing org/ subdirectory",
    )
    parser.add_argument(
        "-v",
        "--vocab",
        required=True,
        dest="vocab_path",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "-b",
        "--bert",
        required=True,
        dest="bert_path",
        help="Path to pretrained BERT model",
    )
    parser.add_argument(
        "-label",
        "--labels",
        required=True,
        dest="label_file",
        help="Path to labels CSV file",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="BERT hidden size (default: 512)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=64, help="Maximum sequence length (default: 64)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.base_dir):
        print(f"ERROR: Base directory not found: {args.base_dir}")
        sys.exit(1)
    if not os.path.exists(args.vocab_path):
        print(f"ERROR: Vocabulary file not found: {args.vocab_path}")
        sys.exit(1)
    if not os.path.exists(args.bert_path):
        print(f"ERROR: BERT model not found: {args.bert_path}")
        sys.exit(1)
    if not os.path.exists(args.label_file):
        print(f"ERROR: Labels file not found: {args.label_file}")
        sys.exit(1)

    # Generate AORGs
    generate_aorgs(
        args.base_dir,
        args.vocab_path,
        args.bert_path,
        args.label_file,
        args.hidden_size,
        args.seq_len,
    )


if __name__ == "__main__":
    main()
