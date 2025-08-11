import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandbprojectname = "linear-probes-location-stable-0-baseline"
saved_activations = "layered_pairs_labels_location_stable_0.pt"
weight_decay_regularization = 0 # change to 1e-2 for regularization

def _make_splits(features, labels, val_size=0.15, test_size=0.15, seed=42):
    y_np = labels.cpu().numpy()
    idx = np.arange(len(y_np))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y_np, test_size=val_size+test_size, random_state=seed, stratify=y_np
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=rel_test, random_state=seed, stratify=y_temp
    )
    def take(idxs):
        x = features[idxs]
        y = labels[idxs]
        return TensorDataset(x, y)
    return take(idx_train), take(idx_val), take(idx_test)

def _eval(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n

def train_linear_probes(
    layered_pairs, layered_labels,
    epochs=2000, batch_size=64, lr=1e-2, seed=42,
    val_size=0.15, test_size=0.15, reverse=True
):
    torch.manual_seed(seed)
    probes, histories, test_results = [], [], []

    order = range(len(layered_pairs)-1, -1, -1) if reverse else range(len(layered_pairs))
    for layer_idx in order:
        pairs  = layered_pairs[layer_idx]
        labels = layered_labels[layer_idx]
        wandb.init(
            project=wandbprojectname,
            name=f"layer_{layer_idx}",
            config={
                "layer": layer_idx,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "val_size": val_size,
                "test_size": test_size,
            }
        )

        features = pairs.sum(dim=1)
        train_set, val_set, test_set = _make_splits(features, labels, val_size, test_size, seed)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

        model = nn.Linear(features.size(1), 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_regularization)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_state = None
        history = []

        for epoch in range(epochs):
            model.train()
            total_loss, correct, n = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                n += x.size(0)

            train_loss = total_loss / n
            train_acc  = correct / n
            val_loss, val_acc = _eval(model, val_loader, loss_fn)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            history.append((train_loss, train_acc, val_loss, val_acc))

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc = _eval(model, test_loader, loss_fn)
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["test_loss"] = test_loss
        wandb.summary["test_accuracy"] = test_acc

        probes.append(model)
        histories.append(history)
        test_results.append((test_loss, test_acc))
        wandb.finish()

    return probes, histories, test_results



def load_pairs(path, device="cpu"):
    obj = torch.load(path, map_location="cpu")
    pairs  = [p.to(device) for p in obj["pairs"]]
    labels = [y.to(device) for y in obj["labels"]]
    return pairs, labels, obj.get("meta", {})

layered_pairs, layered_labels, meta = load_pairs(saved_activations, device="cpu")

probes, histories, test_results = train_linear_probes(layered_pairs, layered_labels)
for i, hist in enumerate(histories):
    tr_loss, tr_acc, vl_loss, vl_acc = hist[-1]
    tl, ta = test_results[i]
    print(f"Layer {11-i}: final train acc {tr_acc:.4f}, val acc {vl_acc:.4f}, test acc {ta:.4f}")
