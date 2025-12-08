#!/usr/bin/env python3
"""
Train GIN (Graph Isomorphism Network) classifier for PDF malware detection

This script trains a GIN model on AORG (Attributed Object Reference Graph) data
with BERT embeddings to classify PDFs as benign or malicious.

Usage:
    python trainGIN.py --train-dir data/train/org_after_prebert \
                       --test-dir data/test/org_after_prebert \
                       --output models/GIN-trained.pth \
                       --epochs 50 \
                       --batch-size 64
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
import dgl
from dgl.nn import GINConv
from dgl.data import DGLDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from pathlib import Path


def setup_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


class OrgDGLDataset(DGLDataset):
    """
    Dataset loader for AORG files (PyTorch Geometric Data -> DGL graphs)

    Loads .pt files containing PyTorch Geometric Data objects and converts
    them to DGL graphs for GIN training.
    """

    def __init__(self, root):
        self.root = root
        super(OrgDGLDataset, self).__init__(name="org_dgl_dataset")
        self.load()

    def load(self):
        """Load all .pt files and convert to DGL graphs"""
        self.graphs = []
        self.labels = []

        print(f"[Dataset] Loading from: {self.root}")

        pt_files = [
            f
            for f in os.listdir(self.root)
            if f.endswith(".pt") and f not in ["pre_filter.pt", "pre_transform.pt"]
        ]

        print(f"[Dataset] Found {len(pt_files)} .pt files")

        for filename in tqdm(pt_files, desc="Loading graphs"):
            file_path = os.path.join(self.root, filename)

            try:
                # Load PyTorch Geometric Data object
                pyg_graph = torch.load(file_path)

                # Extract edges
                if (
                    pyg_graph.edge_index.size(0) > 0
                    and pyg_graph.edge_index.size(1) > 0
                ):
                    src, dst = pyg_graph.edge_index
                else:
                    src, dst = [], []

                # Create DGL graph
                dgl_graph = dgl.graph((src, dst), num_nodes=pyg_graph.x.shape[0])

                # Add node features
                if hasattr(pyg_graph, "x"):
                    dgl_graph.ndata["feat"] = pyg_graph.x.float()

                # Add edge features
                if hasattr(pyg_graph, "edge_attr") and len(pyg_graph.edge_attr) > 0:
                    dgl_graph.edata["feat"] = pyg_graph.edge_attr.float()

                # Store graph and label
                self.graphs.append(dgl_graph)
                self.labels.append(int(pyg_graph.y.item()))

            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")

        print(f"[Dataset] Successfully loaded {len(self.graphs)} graphs")

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def collate(samples):
    """Collate function for batching DGL graphs"""
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = default_collate(labels)
    return batched_graph, batched_labels


class GIN(nn.Module):
    """
    Graph Isomorphism Network for graph classification

    Architecture:
    - 2 GIN convolutional layers
    - Global max pooling
    - Linear classifier
    """

    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(in_feats, hidden_feats), "mean")
        self.conv2 = GINConv(nn.Linear(hidden_feats, hidden_feats), "mean")
        self.classify = nn.Linear(hidden_feats, num_classes)

    def forward(self, g, in_feat):
        # GIN layer 1
        h = self.conv1(g, in_feat)
        h = F.relu(h)

        # GIN layer 2
        h = self.conv2(g, h)

        # Global pooling
        g.ndata["h"] = h
        hg = dgl.max_nodes(g, "h")

        # Classification
        return self.classify(hg)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    for batched_graph, labels in tqdm(train_loader, desc="Training", leave=False):
        batched_graph = batched_graph.to(device)
        labels = labels.to(device).long()
        features = batched_graph.ndata["feat"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(batched_graph, features)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(y_true, y_pred)

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batched_graph, labels in tqdm(val_loader, desc="Validating", leave=False):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).long()
            features = batched_graph.ndata["feat"].to(device)

            # Forward pass
            logits = model(batched_graph, features)
            loss = criterion(logits, labels)

            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


def train_model(
    train_dir, test_dir, output_path, epochs, batch_size, lr, weight_decay, hidden_size
):
    """Main training function"""

    print("=" * 80)
    print("GIN TRAINING - PDF MALWARE DETECTION")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load datasets
    print(f"\n[1/5] Loading datasets...")
    train_dataset = OrgDGLDataset(root=train_dir)
    test_dataset = OrgDGLDataset(root=test_dir)

    print(f"[Dataset] Train size: {len(train_dataset)}")
    print(f"[Dataset] Test size:  {len(test_dataset)}")

    # Get feature dimension from first graph
    feature_dim = train_dataset[0][0].ndata["feat"].shape[1]
    num_classes = 2  # Binary classification

    print(f"[Dataset] Feature dim: {feature_dim}")
    print(f"[Dataset] Num classes: {num_classes}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,  # DGL graphs don't work well with multiprocessing
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # Initialize model
    print(f"\n[2/5] Initializing GIN model...")
    print(
        f"[Model] Architecture: GIN(in={feature_dim}, hidden={hidden_size}, out={num_classes})"
    )

    model = GIN(in_feats=feature_dim, hidden_feats=hidden_size, num_classes=num_classes)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,}")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[3/5] Training configuration:")
    print(f"[Config] Epochs: {epochs}")
    print(f"[Config] Batch size: {batch_size}")
    print(f"[Config] Learning rate: {lr}")
    print(f"[Config] Weight decay: {weight_decay}")

    # Training loop
    print(f"\n[4/5] Starting training...")
    best_val_acc = 0.0
    best_model_state = None
    train_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, test_loader, criterion, device
        )

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(
            f"Val Prec:   {val_prec:.4f} | Val Rec:   {val_rec:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model! (Val Acc: {val_acc:.4f})")

        # Record history
        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        )

    # Save best model
    print(f"\n[5/5] Saving model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save model state
    torch.save(best_model_state, output_path)
    print(f"[Save] Model saved to: {output_path}")

    # Save training history
    history_path = output_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"[Save] Training history saved to: {history_path}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train GIN classifier for PDF malware detection"
    )
    parser.add_argument("--train-dir", required=True, help="Training AORG directory")
    parser.add_argument("--test-dir", required=True, help="Test AORG directory")
    parser.add_argument("--output", required=True, help="Output model path (.pth)")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Hidden layer size (default: 256)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.train_dir):
        print(f"ERROR: Train directory not found: {args.train_dir}")
        sys.exit(1)
    if not os.path.exists(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Train model
    train_model(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
    )


if __name__ == "__main__":
    main()
