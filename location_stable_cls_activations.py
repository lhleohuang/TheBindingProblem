import json
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.utils import prisma_utils

MODE = 0 # 1 = No location swapping in SAME pairs, 0 = Swapping locations for SAME pairs

dataset_name = f"bag_dataset_location_stable_{MODE}"
DEVICE     = "cpu"
out_path = f"layered_pairs_labels_location_stable_{MODE}.pt"

tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])

class FlatImageDataset(Dataset):
    """
    Flattens the pair dataset into single images.
    Preserves:
      - same_pairs_idx: list of (idx_img0, idx_img1) for 'same' pairs
      - diff_pairs_idx: list of (idx_img0, idx_img1) for 'different' pairs
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.images = []          # image file paths
        self.labels = []          # 0 = same, 1 = different
        self.meta_per_image = []  # metadata dict per image

        self.same_pairs_idx = []
        self.diff_pairs_idx = []

        def _ingest(meta_path, label_value, pair_list):
            with open(meta_path) as f:
                for row in f:
                    js = json.loads(row)
                    start_idx = len(self.images)

                    # image 0
                    self.images.append(js["image0"])
                    self.labels.append(label_value)
                    self.meta_per_image.append({
                        "pair_id": js["pair_id"],
                        "which": 0,
                        "label": label_value,
                        "s1": js.get("s1"), "c1": js.get("c1"),
                        "s2": js.get("s2"), "c2": js.get("c2")
                    })

                    # image 1
                    self.images.append(js["image1"])
                    self.labels.append(label_value)
                    self.meta_per_image.append({
                        "pair_id": js["pair_id"],
                        "which": 1,
                        "label": label_value,
                        "s1": js.get("s1"), "c1": js.get("c1"),
                        "s2": js.get("s2"), "c2": js.get("c2")
                    })

                    # store the tuple of indices for this pair
                    pair_list.append((start_idx, start_idx + 1))

        same_meta = self.root / "same_metadata.jsonl"
        diff_meta = self.root / "different_metadata.jsonl"

        if same_meta.exists():
            _ingest(same_meta, 0, self.same_pairs_idx)
        if diff_meta.exists():
            _ingest(diff_meta, 1, self.diff_pairs_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, path

# Helper for plotting stats
def _bar_triptych(cnt_shapes, cnt_colors, cnt_combo, title, save_dir=None):
    def kv(cnt):
        items = sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0])))
        x = [str(k) for k,_ in items]
        y = [v for _,v in items]
        return x, y

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    xs, ys = kv(cnt_shapes)
    axes[0].bar(range(len(xs)), ys)
    axes[0].set_title("Shapes")
    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels(xs, rotation=60, ha="right")

    xc, yc = kv(cnt_colors)
    axes[1].bar(range(len(xc)), yc)
    axes[1].set_title("Colors")
    axes[1].set_xticks(range(len(xc)))
    axes[1].set_xticklabels(xc, rotation=60, ha="right")

    xk, yk = kv(cnt_combo)
    axes[2].bar(range(len(xk)), yk)
    axes[2].set_title("Shape+Color")
    axes[2].set_xticks(range(len(xk)))
    axes[2].set_xticklabels(xk, rotation=60, ha="right")

    fig.suptitle(title)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p = Path(save_dir) / f"{title.lower().replace(' ','_').replace('+','plus')}.png"
        fig.savefig(p, dpi=200)
    plt.show()

def get_pair_stats(dataset: FlatImageDataset):
    color_counts_diff = Counter()
    shape_counts_diff = Counter()
    shape_color_counts_diff = Counter()
    color_counts_same = Counter()
    shape_counts_same = Counter()
    shape_color_counts_same = Counter()

    for idx0, idx1 in dataset.diff_pairs_idx:
        for idx in (idx0, idx1):
            meta = dataset.meta_per_image[idx]
            # figure out the object's shape/color for this image index
            # 'which' tells us whether this is first or second object in the pair
            if meta["which"] == 0:
                shape = meta["s1"]
                color = meta["c1"]
            else:
                shape = meta["s2"]
                color = meta["c2"]

            color_counts_diff[color] += 1
            shape_counts_diff[shape] += 1
            shape_color_counts_diff[(shape, color)] += 1

    for idx0, idx1 in dataset.same_pairs_idx:
        for idx in (idx0, idx1):
            meta = dataset.meta_per_image[idx]
            # figure out the object's shape/color for this image index
            # 'which' tells us whether this is first or second object in the pair
            if meta["which"] == 0:
                shape = meta["s1"]
                color = meta["c1"]
            else:
                shape = meta["s2"]
                color = meta["c2"]

            color_counts_same[color] += 1
            shape_counts_same[shape] += 1
            shape_color_counts_same[(shape, color)] += 1

    print(f"SAME pairs stats:\n")
    print(f"Colors (total={sum(color_counts_same.values())}, unique={len(color_counts_same)}):")
    for color, count in sorted(color_counts_same.items()):
        print(f"  {color}: {count}")

    print(f"\nShapes (total={sum(shape_counts_same.values())}, unique={len(shape_counts_same)}):")
    for shape, count in sorted(shape_counts_same.items()):
        print(f"  {shape}: {count}")

    print(f"\nShape+Color (total={sum(shape_color_counts_same.values())}, unique={len(shape_color_counts_same)}):")
    for sc, count in sorted(shape_color_counts_same.items()):
        print(f"  {sc}: {count}")

    print(f"\nDIFFERENT pairs stats:\n")
    print(f"Colors (total={sum(color_counts_diff.values())}, unique={len(color_counts_diff)}):")
    for color, count in sorted(color_counts_diff.items()):
        print(f"  {color}: {count}")

    print(f"\nShapes (total={sum(shape_counts_diff.values())}, unique={len(shape_counts_diff)}):")
    for shape, count in sorted(shape_counts_diff.items()):
        print(f"  {shape}: {count}")

    print(f"\nShape+Color (total={sum(shape_color_counts_diff.values())}, unique={len(shape_color_counts_diff)}):")
    for sc, count in sorted(shape_color_counts_diff.items()):
        print(f"  {sc}: {count}")

    _bar_triptych(shape_counts_same, color_counts_same, shape_color_counts_same,
                  f"location_stable_{MODE}_same_stats", save_dir="dataset_stats_plots")
    _bar_triptych(shape_counts_diff, color_counts_diff, shape_color_counts_diff,
                  f"location_stable_{MODE}_diff_stats", save_dir="dataset_stats_plots")

ds = FlatImageDataset(dataset_name, transform=tfm)
get_pair_stats(ds)
loader = DataLoader(ds, batch_size=100000, shuffle=False, num_workers=0) # include all images in one batch

same_pairs_idx = ds.same_pairs_idx
diff_pairs_idx = ds.diff_pairs_idx

print("Num images:", len(ds))
print("Num same pairs:", len(ds.same_pairs_idx))
print("Num diff pairs:", len(ds.diff_pairs_idx))
print("First same pair indices:", ds.same_pairs_idx[0])
print("First diff pair indices:", ds.diff_pairs_idx[0])

# ----------------------------
# Encode images with DINO (CLS activations per layer)
# ----------------------------
model = load_hooked_model(
    'facebook/dino-vitb16',
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True
).to(DEVICE)

# We’ll accumulate CLS per layer across batches
layered_cls_list = None  # will become [list_of_tensors_per_layer]
with torch.no_grad():
    for xb, _paths in loader:
        xb = xb.to(DEVICE)
        _logits, cache = model.run_with_cache(xb)
        # grab CLS for each layer
        cls_batch_per_layer = [cache[prisma_utils.get_act_name("resid_mid", l, None)][:, 0, :].cpu()
                               for l in range(model.cfg.n_layers)]
        if layered_cls_list is None:
            layered_cls_list = [t for t in cls_batch_per_layer]
        else:
            for l in range(len(layered_cls_list)):
                layered_cls_list[l] = torch.cat([layered_cls_list[l], cls_batch_per_layer[l]], dim=0)

print("Collected CLS activations for", len(layered_cls_list), "layers.")

# ----------------------------
# Build pair tensors directly from metadata-defined pairs
# ----------------------------
def build_pairs_from_indices(layered_cls, same_pairs_idx, diff_pairs_idx):
    layered_same, layered_diff = [], []
    for cls_acts in layered_cls:  # cls_acts: [N, D]
        D = cls_acts.size(-1)
        same_pairs = torch.stack([torch.stack((cls_acts[i], cls_acts[j]), dim=0)
                                  for (i, j) in same_pairs_idx]) if same_pairs_idx else torch.empty(0, 2, D)
        diff_pairs = torch.stack([torch.stack((cls_acts[i], cls_acts[j]), dim=0)
                                  for (i, j) in diff_pairs_idx]) if diff_pairs_idx else torch.empty(0, 2, D)
        layered_same.append(same_pairs)
        layered_diff.append(diff_pairs)
    return layered_same, layered_diff

layered_same, layered_diff = build_pairs_from_indices(layered_cls_list, same_pairs_idx, diff_pairs_idx)

# breakpoint()

# ----------------------------
# Combine, shuffle, save
# ----------------------------
def combine_and_shuffle_pairs(layered_same, layered_diff, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    lp, ll = [], []
    for same_pairs, diff_pairs in zip(layered_same, layered_diff):
        n_same, n_diff = same_pairs.size(0), diff_pairs.size(0)
        y_same = torch.zeros(n_same, dtype=torch.long)
        y_diff = torch.ones(n_diff, dtype=torch.long)
        pairs  = torch.cat([same_pairs, diff_pairs], dim=0)
        labels = torch.cat([y_same, y_diff], dim=0)
        perm = torch.randperm(pairs.size(0))
        lp.append(pairs[perm]); ll.append(labels[perm])
    return lp, ll

def save_pairs(path, layered_pairs, layered_labels, meta=None, fp16=False):
    payload = {
        "version": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "pairs": [p.half().cpu() if fp16 else p.cpu() for p in layered_pairs],
        "labels": [y.cpu() for y in layered_labels],
        "meta": meta or {}
    }
    torch.save(payload, path)

layered_pairs, layered_labels = combine_and_shuffle_pairs(layered_same, layered_diff, seed=42)
meta = {
    "model": "facebook/dino-vitb16",
    "pair_source": "metadata.jsonl",
    "images_dir": str(Path(dataset_name)),
    "seed": 42
}
save_pairs(out_path, layered_pairs, layered_labels, meta=meta, fp16=False)

print(f"Saved pairs to {out_path}")
