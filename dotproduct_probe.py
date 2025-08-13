import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

class PairDotModel(nn.Module):
    def __init__(self, input_dim, proj_dim=None, l2_normalize=True, temperature=0.07):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.temperature = temperature
        self.proj = nn.Linear(input_dim, proj_dim if proj_dim else input_dim, bias=False)
    def encode(self, x):
        x = self.proj(x)
        if self.l2_normalize:
            x = nn.functional.normalize(x, dim=1)
        return x
    def forward(self, x):
        a, b = x[:, 0, :], x[:, 1, :]
        a = self.encode(a)
        b = self.encode(b)
        return a, b

def clip_loss(a, b, temperature=0.07):
    logits = (a @ b.T) / temperature
    targets = torch.arange(a.size(0), device=a.device)
    loss_i = nn.functional.cross_entropy(logits, targets)
    loss_t = nn.functional.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)

@torch.no_grad()
def _pair_scores(model, pairs, batch_size=800, device=None):
    device = device or next(model.parameters()).device
    dl = DataLoader(TensorDataset(pairs.to(device)), batch_size=batch_size, shuffle=False)
    out = []
    for (xb,) in dl:
        a, b = model(xb)
        s = (a * b).sum(dim=1) / model.temperature
        out.append(s.detach().cpu())
    return torch.cat(out, dim=0).numpy()

@torch.no_grad()
def evaluate(model, pairs, labels, batch_size=800):
    scores = _pair_scores(model, pairs, batch_size=batch_size)
    y = labels.float().cpu().numpy()
    y_pos = 1.0 - y
    roc = roc_auc_score(y_pos, scores)
    ap  = average_precision_score(y_pos, scores)
    return {"roc_auc": roc, "ap": ap}

def _stratified_split_indices(labels, val_size=0.2, seed=42):
    g0 = torch.where(labels == 0)[0]
    g1 = torch.where(labels == 1)[0]
    gen = torch.Generator().manual_seed(seed)
    if g0.numel() > 0:
        g0 = g0[torch.randperm(g0.numel(), generator=gen)]
    if g1.numel() > 0:
        g1 = g1[torch.randperm(g1.numel(), generator=gen)]
    n0_val = int(round(val_size * g0.numel()))
    n1_val = int(round(val_size * g1.numel()))
    val_idx = torch.cat([g0[:n0_val], g1[:n1_val]]) if (g0.numel() or g1.numel()) else torch.empty(0, dtype=torch.long)
    train_idx = torch.cat([g0[n0_val:], g1[n1_val:]]) if (g0.numel() or g1.numel()) else torch.empty(0, dtype=torch.long)
    return train_idx, val_idx

def train_val_pairdot_per_layer(
    layered_pairs,
    layered_labels,
    proj_dim=256,
    l2_normalize=True,
    temperature=0.07,
    val_size=0.2,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device=None,
    progress=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models, logs = [], []
    for layer_idx, (pairs, labels) in enumerate(zip(layered_pairs, layered_labels)):
        train_idx, val_idx = _stratified_split_indices(labels, val_size=val_size, seed=42)
        pairs_train, labels_train = pairs[train_idx].to(device), labels[train_idx].to(device)
        pairs_val,   labels_val   = pairs[val_idx].to(device), labels[val_idx].to(device)
        train_dl = DataLoader(TensorDataset(pairs_train, labels_train), batch_size=batch_size, shuffle=True)
        model = PairDotModel(
            input_dim=pairs.size(-1),
            proj_dim=proj_dim,
            l2_normalize=l2_normalize,
            temperature=temperature,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for xb, yb in train_dl:
                pos = (yb == 0)
                if pos.sum() < 2:
                    continue
                a, b = model(xb[pos])
                loss = clip_loss(a, b, temperature=model.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
        tr = evaluate(model, pairs_train, labels_train, batch_size=batch_size)
        va = evaluate(model, pairs_val, labels_val, batch_size=batch_size)
        if progress:
            print(f"Layer {layer_idx}: Train AUC {tr['roc_auc']:.4f} (AP {tr['ap']:.4f}) | Val AUC {va['roc_auc']:.4f} (AP {va['ap']:.4f})")
        models.append(model)
        logs.append({"train": tr, "val": va})
    return models, logs

def load_pairs(path, device="cpu"):
    obj = torch.load(path, map_location="cpu")
    pairs  = [p.to(device) for p in obj["pairs"]]
    labels = [y.to(device) for y in obj["labels"]]
    return pairs, labels, obj.get("meta", {})

def parse_label(fname):
    if fname.endswith(".pt") and "layered_pairs_labels_" in fname:
        return fname[len("layered_pairs_labels_"):-len(".pt")]
    return os.path.splitext(os.path.basename(fname))[0]

def train_and_plot_on_files(
    saved_activation_files,
    proj_dim=256,
    l2_normalize=True,
    temperature=0.07,
    val_size=0.2,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    device="cpu",
    out_dir="trained_plots"
):
    os.makedirs(out_dir, exist_ok=True)
    perfile_train_aucs = {}
    perfile_val_aucs = {}
    for fname in saved_activation_files:
        layered_pairs, layered_labels, _ = load_pairs(fname, device=device)
        # breakpoint()
        models, logs = train_val_pairdot_per_layer(
            layered_pairs, layered_labels,
            proj_dim=proj_dim, l2_normalize=l2_normalize, temperature=temperature,
            val_size=val_size, epochs=epochs, batch_size=batch_size, lr=lr, device=device, progress=False
        )
        train_aucs = [entry["train"]["roc_auc"] for entry in logs]
        val_aucs   = [entry["val"]["roc_auc"]   for entry in logs]
        label = parse_label(fname)
        perfile_train_aucs[label] = train_aucs
        perfile_val_aucs[label] = val_aucs
        print(f"{label}: train AUC mean={sum(train_aucs)/len(train_aucs):.4f}, val AUC mean={sum(val_aucs)/len(val_aucs):.4f}")
    L = len(next(iter(perfile_train_aucs.values())))
    layers = list(range(L))
    plt.figure(figsize=(8,5))
    for label, aucs in perfile_train_aucs.items():
        plt.plot(layers, aucs, marker="o", label=label)
    plt.title("Train ROC–AUC per Layer")
    plt.xlabel("Layer")
    plt.ylabel("ROC–AUC")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8)
    train_path = os.path.join(out_dir, "train_roc_auc_per_layer.png")
    plt.savefig(train_path, dpi=200)
    plt.close()
    plt.figure(figsize=(8,5))
    for label, aucs in perfile_val_aucs.items():
        plt.plot(layers, aucs, marker="o", label=label)
    plt.title("Val ROC–AUC per Layer")
    plt.xlabel("Layer")
    plt.ylabel("ROC–AUC")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8)
    val_path = os.path.join(out_dir, "val_roc_auc_per_layer.png")
    plt.savefig(val_path, dpi=200)
    plt.close()
    print(f"Saved plots:\n  {train_path}\n  {val_path}")

# Example
saved_activation_files = [
    "layered_pairs_labels_strict_diff.pt",
    "layered_pairs_labels_one_fixed_one_changes_one_feature.pt",
    "layered_pairs_labels_location_stable_0.pt",
    "layered_pairs_labels_location_stable_1.pt",
    "layered_pairs_labels_superposition_catastrophe.pt",
]

train_and_plot_on_files(
    saved_activation_files,
    proj_dim=None,
    l2_normalize=True,
    temperature=0.07,
    val_size=0.2,
    epochs=200,
    batch_size=128,
    lr=1e-3,
    device="cpu",
    out_dir="dot_product_probe_plots"
)
