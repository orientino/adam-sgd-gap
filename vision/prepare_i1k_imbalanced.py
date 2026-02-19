from pathlib import Path

import numpy as np
import webdataset as wds
from tqdm import tqdm

DIR_DATA = "/project/home/p200535/data"
N_CLASSES = 1000
CLASS_COUNTS = np.array([int(np.ceil(1300 / (i + 1))) for i in range(N_CLASSES)], dtype=np.int64)  # fmt: skip


def load_i1k_parts(dir_data):
    base = Path(dir_data) / "imagenet1k"
    dataset = wds.WebDataset(
        str(base / "imagenet1k-train-{0000..1023}.tar"),
        shardshuffle=False,
    )
    for sample in dataset:
        yield sample["jpg"], int(sample["cls"])


def create_i1k_imbalanced(dir_data):
    rng = np.random.default_rng(42)

    class_totals = np.zeros(N_CLASSES, dtype=np.int64)
    for _, y in tqdm(load_i1k_parts(dir_data), total=1281167):
        class_totals[y] += 1

    sorted_classes = np.argsort(-class_totals)
    target_counts = np.zeros(N_CLASSES, dtype=np.int64)
    target_counts[sorted_classes] = CLASS_COUNTS
    target_counts = np.minimum(target_counts, class_totals)
    sampled_positions = [
        np.sort(rng.choice(class_totals[c], size=target_counts[c], replace=False))
        for c in range(N_CLASSES)
    ]

    out_dir = Path(dir_data) / "imagenet1k-imbalanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_target = int(target_counts.sum())
    out_x = np.empty(total_target, dtype=object)
    out_y = np.empty(total_target, dtype=np.int64)

    seen_per_class = np.zeros(N_CLASSES, dtype=np.int64)
    ptr_per_class = np.zeros(N_CLASSES, dtype=np.int64)
    write = 0

    for x, y in tqdm(load_i1k_parts(dir_data), total=total_target):
        pos = seen_per_class[y]
        seen_per_class[y] += 1

        ptr = ptr_per_class[y]
        picked = sampled_positions[y]
        if ptr < len(picked) and picked[ptr] == pos:
            out_x[write] = x
            out_y[write] = y
            ptr_per_class[y] += 1
            write += 1

    out_path = out_dir / "i1k_imbalanced.npz"
    np.savez(out_path, X=out_x, Y=out_y)

    counts = np.bincount(out_y, minlength=N_CLASSES)
    print("Saved:", out_path)
    print("Shape X:", out_x.shape, "Shape Y:", out_y.shape)
    print("Class counts:", np.sort(counts)[::-1])
    print("Total:", int(counts.sum()))
    assert counts.sum() == CLASS_COUNTS.sum()


create_i1k_imbalanced(DIR_DATA)
