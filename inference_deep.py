import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from data import MIOSTONEDataset, MIOSTONETree
from train import TrainingPipeline

from model import MIOSTONEModel
from baseline import MLP, PopPhyCNN, TaxoNN, MDeep, DeepBiome, PhCNN

def run_inference(model_fp, results_fp, data_fp, meta_fp, target_fp, tree_fp):
    # 1. load model type from results file
    with open(results_fp) as f:
        result_json = json.load(f)
    model_type = result_json["Model Type"].lower()

    # 2. initialization, load data and tree
    pipeline = TrainingPipeline(seed=42)
    pipeline._load_tree(tree_fp)

    # Disable CLR preprocessing for PopPhyCNN
    preprocess = False if model_type == "popphycnn" else True
    pipeline._load_data(data_fp, meta_fp, target_fp, preprocess=preprocess)

    # 3. load model and hyperparameters
    pipeline._load_model(model_fp, results_fp)
    model = pipeline.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 4. reshape data for PopPhyCNN if needed
    if model_type == "popphycnn":
        pipeline.data.to_popphycnn_matrix(pipeline.tree)

    # 5. build DataLoader
    dataset = pipeline.data
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    # 6. perform inference
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device).to(torch.float32)  # ensure input type
            logits = model(X)
            all_logits.append(logits.cpu())
            all_labels.append(y)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 7. calculate evaluation metrics using sklearn
    probs = F.softmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    if probs.shape[1] == 2:
        auroc = roc_auc_score(labels_np, probs[:, 1])
        auprc = average_precision_score(labels_np, probs[:, 1])
    else:
        auroc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
        auprc = average_precision_score(
            F.one_hot(all_labels, num_classes=probs.shape[1]).numpy(),
            probs,
            average="macro",
        )

    print(f"[{model_type.upper()}] Test AUROC: {auroc:.4f}")
    print(f"[{model_type.upper()}] Test AUPRC: {auprc:.4f}")

    # 8. optional: save logits + labels
    # np.savez("inference_outputs.npz", logits=all_logits.numpy(), labels=all_labels.numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using a trained MIOSTONE model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model .pt file")
    parser.add_argument('--results', type=str, required=True, help="Path to the training results .json file")
    parser.add_argument('--data', type=str, required=True, help="Path to data.tsv.xz")
    parser.add_argument('--meta', type=str, required=True, help="Path to meta.tsv")
    parser.add_argument('--target', type=str, required=True, help="Path to target.py")
    parser.add_argument('--tree', type=str, required=True, help="Path to tree.nwk")

    args = parser.parse_args()

    run_inference(
        model_fp=args.model,
        results_fp=args.results,
        data_fp=args.data,
        meta_fp=args.meta,
        target_fp=args.target,
        tree_fp=args.tree
    )
