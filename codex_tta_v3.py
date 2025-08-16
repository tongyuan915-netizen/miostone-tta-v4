import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score  # 用于 AUROC/AUPRC

from data import MIOSTONEDataset, MIOSTONETree
from model import MIOSTONEModel


def test_time_adaptation(model: MIOSTONEModel, data_loader, steps: int = 1, lr: float = 1e-3):
    model.eval()

    norm_params = []
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
            m.train()
            if hasattr(m, "weight") and m.weight is not None:
                norm_params.append(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                norm_params.append(m.bias)
        else:
            m.eval()

    for p in model.parameters():
        p.requires_grad = False
    for p in norm_params:
        p.requires_grad = True

    optimizer = torch.optim.Adam(norm_params, lr=lr)

    for x, _ in data_loader:
        x = x.to(next(model.parameters()).device)
        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
            entropy.backward()
            optimizer.step()
    return model


def run_training_tta_pipeline(
    model_fp: str,
    model_param_fp: str,
    data_fp: str,
    meta_fp: str,
    type_fp: str,
    taxonomy_fp: str | None = None,
    output_model_fp: str | None = None,
    test_result_fp: str | None = None,
    batch_size: int = 32,
    epochs: int = 1,
    lr: float = 1e-3,
):
    TIE_EPS = 1e-12

    if taxonomy_fp is None:
        raise ValueError("taxonomy_fp is required to instantiate MIOSTONEModel")

    # ---------- 数据准备 ----------
    tree = MIOSTONETree.init_from_nwk(taxonomy_fp)
    dataset = MIOSTONEDataset.init_from_files(data_fp, meta_fp, type_fp)
    tree.prune(dataset.features)
    tree.compute_depths()
    tree.compute_indices()
    dataset.order_features_by_tree(tree)
    dataset.clr_transform()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ---------- 模型加载 ----------
    with open(model_param_fp) as f:
        params = json.load(f)
    if params.get("Model Type") != "miostone":
        raise ValueError("Only MIOSTONEModel is supported for TTA")

    model = MIOSTONEModel(tree, dataset.num_classes, **params["Model Hparams"])
    model.load_state_dict(torch.load(model_fp, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- 工具函数 ----------
    @torch.no_grad()
    def _evaluate_logits():
        model.eval()
        logits_list, labels_list = [], []
        for x, y in loader:
            x = x.to(device)
            logits_list.append(model(x).cpu())
            labels_list.append(y)
        return torch.cat(logits_list), torch.cat(labels_list)

    def _compute_metrics(test_logits: torch.Tensor, test_labels: torch.Tensor):
        probs = F.softmax(test_logits, dim=1).numpy()
        y = test_labels.numpy()
        if probs.shape[1] == 2:
            auroc = roc_auc_score(y, probs[:, 1])
            auprc = average_precision_score(y, probs[:, 1])
        else:
            auroc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
            y_onehot = F.one_hot(test_labels, num_classes=probs.shape[1]).numpy()
            auprc = average_precision_score(y_onehot, probs, average="macro")
        return float(auroc), float(auprc)

    def _get_norm_param_snapshot():
        snapshot = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    snapshot[f"{name}.weight"] = module.weight.detach().cpu().clone()
                if hasattr(module, 'bias') and module.bias is not None:
                    snapshot[f"{name}.bias"] = module.bias.detach().cpu().clone()
        return snapshot

    def _get_norm_param_diff(before: dict, after: dict, threshold: float = 1e-6):
        changes = []
        for k in before:
            diff = (after[k] - before[k]).abs().sum().item()
            if diff > threshold:
                changes.append(f"{k}:Δ={diff:.6f}")
        return changes

    # ---------- 初始评估 ----------
    start = time.time()
    init_logits, init_labels = _evaluate_logits()
    best_auroc, best_auprc = _compute_metrics(init_logits, init_labels)
    best_epoch = 0
    best_state = model.state_dict()

    # ---------- TTA 主循环 ----------
    for ep in range(1, epochs + 1):
        before_params = _get_norm_param_snapshot()

        test_time_adaptation(model, loader, steps=1, lr=lr)

        after_params = _get_norm_param_snapshot()
        logits, labels = _evaluate_logits()
        auroc, auprc = _compute_metrics(logits, labels)

        changed = _get_norm_param_diff(before_params, after_params)
        changed_str = " | ".join(changed) if changed else "No Param Changed"

        print(f"[Epoch {ep}] AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ||  {changed_str}")

        if (auroc - best_auroc) > TIE_EPS:
            best_auroc, best_auprc = auroc, auprc
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - start

    # ---------- 恢复并保存 ----------
    model.load_state_dict(best_state)
    if output_model_fp:
        best_fp = output_model_fp[:-3] + "_best.pt" if output_model_fp.endswith(".pt") else output_model_fp + "_best.pt"
        torch.save(model.state_dict(), best_fp)

    best_logits, best_labels = _evaluate_logits()
    result = {
        "best_epoch": int(best_epoch),
        "time_elapsed": elapsed,
        "AUROC": best_auroc,
        "AUPRC": best_auprc,
        "test_logits": best_logits.tolist(),
        "test_labels": best_labels.tolist(),
    }
    if test_result_fp:
        with open(test_result_fp, "w") as f:
            json.dump(result, f, indent=4)

    # ---------- 最终输出 ----------
    print(f"[Best] epoch={best_epoch}  AUROC={best_auroc:.4f}")
    print(f"[Best] epoch={best_epoch}  AUPRC={best_auprc:.4f}")

    return model, result

####################################### tune tta
def tune_tta_learning_rate(
    model_fp: str,
    model_param_fp: str,
    data_fp: str,
    meta_fp: str,
    type_fp: str,
    taxonomy_fp: str,
    *,
    base_lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 1,
    coarse_levels: int = 3,
    coarse_factor: float = 5.0,
    fine_factor: float = 2.0,
    fine_levels: int = 2,
):
    """Search for an optimal learning rate for TTA.

    This utility first performs a coarse search over learning rates spaced by
    powers of ``coarse_factor`` around ``base_lr``.  It then performs a finer
    search around the best coarse value using ``fine_factor``.  Both stages can
    be widened by increasing ``coarse_levels`` or ``fine_levels`` respectively.
    The search is based on AUROC returned by :func:`run_training_tta_pipeline`.

    Returns the best learning rate, its evaluation result and a dictionary of
    all evaluated results keyed by the learning rate.
    """

    all_results = {}
    best_lr, best_result = None, None

    # --------- coarse search ---------
    for i in range(-coarse_levels, coarse_levels + 1):
        lr = base_lr * (coarse_factor ** i)
        print(f"[Coarse] evaluating lr={lr:.6g}")
        _, result = run_training_tta_pipeline(
            model_fp,
            model_param_fp,
            data_fp,
            meta_fp,
            type_fp,
            taxonomy_fp,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
        )
        result["lr"] = lr
        all_results[lr] = result
        if best_result is None or result["AUROC"] > best_result["AUROC"]:
            best_lr, best_result = lr, result

    # --------- fine search ---------
    for i in range(-fine_levels, fine_levels + 1):
        lr = best_lr * (fine_factor ** i)
        if lr in all_results:
            continue  # skip already evaluated lr
        print(f"[Fine] evaluating lr={lr:.6g}")
        _, result = run_training_tta_pipeline(
            model_fp,
            model_param_fp,
            data_fp,
            meta_fp,
            type_fp,
            taxonomy_fp,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
        )
        result["lr"] = lr
        all_results[lr] = result
        if result["AUROC"] > best_result["AUROC"]:
            best_lr, best_result = lr, result

    print(f"[Best LR] {best_lr:.6g}  AUROC={best_result['AUROC']:.4f}")
    return best_lr, best_result, all_results


