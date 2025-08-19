import numpy as np
import os
import torch
import random
from datetime import datetime
from vit_prisma.models.model_loader import load_hooked_model
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from PIL import Image
from vit_prisma.utils import prisma_utils
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Tuple
import torchvision.transforms as T

class Bag3ImageDataset(Dataset):
    """
    Loads SINGLE images from the generated bag-of-3 dataset.
    - Preserves ordering exactly as listed in metadata.jsonl (pair_00000 img_0, pair_00000 img_1, pair_00001 img_0, ...)
    - Does NOT shuffle and does NOT return paired samples.
    - You can later reconstruct pairs using the provided meta fields (pair_id, image_idx, label).
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,  # None = load all; else "train" | "val" | "test"
        transform: Optional[Any] = None,
        return_meta: bool = True,
    ):
        """
        Args:
            root_dir: Path to dataset root (the directory that contains train/, val/, test/, and metadata.jsonl)
            split: If provided, only load items with this split ("train", "val", or "test"). If None, load all.
            transform: torchvision transform applied to images. If None, a default 224x224 ToTensor/Normalize is used.
            return_meta: If True, __getitem__ returns (image_tensor, meta_dict). If False, returns only image_tensor.
        """
        self.root = Path(root_dir)
        self.meta_path = self.root / "metadata.jsonl"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found at {self.meta_path}")

        self.split = split
        self.return_meta = return_meta

        # Default transform matches common ViT preprocessing (tweak as needed)
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform

        # Load metadata lines in-order and optionally filter by split
        self.items: List[Dict[str, Any]] = []
        with open(self.meta_path, "r") as f:
            for line in f:
                row = json.loads(line)
                if self.split is None or row["split"] == self.split:
                    self.items.append(row)

        # Build quick lookup helpers if you need to reconstruct pairs later
        # pair_id -> [dataset_index_of_img0, dataset_index_of_img1] (order preserved if both present in this split)
        self.pair_id_to_indices: Dict[int, List[int]] = {}
        for idx, row in enumerate(self.items):
            pid = int(row["pair_id"])
            self.pair_id_to_indices.setdefault(pid, []).append(idx)

        # Optional convenience lists (only include pairs fully contained in this dataset instance)
        self.same_pairs_idx: List[Tuple[int, int]] = []
        self.diff_pairs_idx: List[Tuple[int, int]] = []
        for pid, idxs in self.pair_id_to_indices.items():
            if len(idxs) == 2:
                i0, i1 = sorted(idxs, key=lambda k: self.items[k]["image_idx"])
                lbl = self.items[i0]["label"]
                if lbl == "same":
                    self.same_pairs_idx.append((i0, i1))
                elif lbl == "diff":
                    self.diff_pairs_idx.append((i0, i1))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        row = self.items[idx]
        # path in metadata is like "train/pair_00000__same__img_0.png"
        img_path = self.root / row["file"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x = self.transform(im) if self.transform is not None else im

        if self.return_meta:
            # Label per *pair*; you’ll restructure later
            meta = {
                "pair_id": int(row["pair_id"]),
                "image_idx": int(row["image_idx"]),     # 0 or 1
                "label_str": row["label"],              # "same" or "diff"
                "split": row["split"],                  # "train" | "val" | "test"
                "file": row["file"],                    # relative path
                # Optional: the original bags (useful for analysis)
                "bag_shapes": row.get("bag_shapes"),
                "bag_colors": row.get("bag_colors"),
                "objects": row.get("objects"),          # [{shape,color,xy}, ...]
            }
            return x, meta
        else:
            return x


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    model = load_hooked_model('facebook/dino-vitb16', center_writing_weights=True,
                                        fold_ln=True,
                                        refactor_factored_attn_matrices=True, device="cpu")

    ds_train = Bag3ImageDataset("bag3_dataset", split="train", return_meta=True)
    ds_val   = Bag3ImageDataset("bag3_dataset", split="val",   return_meta=True)
    ds_test  = Bag3ImageDataset("bag3_dataset", split="test",  return_meta=True)

    # IMPORTANT: do not shuffle to preserve ordering
    train_loader = DataLoader(ds_train, batch_size=100000, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=100000, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=100000, shuffle=False, num_workers=4, pin_memory=True)

    # Quick sanity check on ordering and metadata:
    xb, meta = next(iter(train_loader))
    print(xb.shape)  # [B, 3, 224, 224]
    device = torch.device("cpu")
    xb = xb.to(device)
    model = model.to(device)

    all_logits, all_cache = model.run_with_cache(xb)

    layered_activations = []

    for layer in range(model.cfg.n_layers):
        layered_activations.append(all_cache[prisma_utils.get_act_name("resid_mid", layer, None)])

    print(layered_activations[0].shape)  # [B, 197, 768] for ViT-B/16

