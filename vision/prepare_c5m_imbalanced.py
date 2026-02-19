from pathlib import Path

import numpy as np

DIR_DATA = "/project/home/p200535/data"
CLASS_COUNTS = np.array([500_000 // (i + 1) for i in range(10)], dtype=np.int64)


def load_c5m_parts(dir_data):
    base = Path(dir_data) / "cifar-5m"
    for i in range(6):
        with np.load(base / f"part{i}.npz") as data:
            yield data["X"], data["Y"]


def create_c5m_imbalanced(dir_data):
    rng = np.random.default_rng(42)

    class_totals = np.zeros(10, dtype=np.int64)
    for _, y in load_c5m_parts(dir_data):
        class_totals += np.bincount(y, minlength=10)

    sampled_positions = [
        np.sort(rng.choice(class_totals[c], size=CLASS_COUNTS[c], replace=False))
        for c in range(10)
    ]

    out_dir = Path(dir_data) / "cifar-5m-imbalanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_target = int(CLASS_COUNTS.sum())
    x_tmp = out_dir / "c5m_imbalanced_X.tmp.npy"
    y_tmp = out_dir / "c5m_imbalanced_Y.tmp.npy"

    out_x = None
    out_y = np.lib.format.open_memmap(
        y_tmp, mode="w+", dtype=np.int64, shape=(total_target,)
    )

    seen_per_class = np.zeros(10, dtype=np.int64)
    write = 0

    for part_idx, (x, y) in enumerate(load_c5m_parts(dir_data)):
        local_pick = []
        for c in range(10):
            class_idx = np.flatnonzero(y == c)
            n_class = len(class_idx)
            picked = sampled_positions[c]
            start = seen_per_class[c]
            stop = start + n_class
            rel = picked[(picked >= start) & (picked < stop)] - start
            if len(rel):
                local_pick.append(class_idx[rel])
            seen_per_class[c] = stop

        if not local_pick:
            continue

        local_pick = np.concatenate(local_pick).astype(np.int64)
        n = len(local_pick)

        if out_x is None:
            out_x = np.lib.format.open_memmap(
                x_tmp, mode="w+", dtype=x.dtype, shape=(total_target, *x.shape[1:])
            )

        out_x[write : write + n] = x[local_pick]
        out_y[write : write + n] = y[local_pick]
        write += n
        print(f"Processed part {part_idx}: selected {n}")

    out_x.flush()
    out_y.flush()
    counts = np.bincount(out_y, minlength=10)
    x_shape = out_x.shape

    out_path = out_dir / "c5m_imbalanced.npz"
    np.savez(out_path, X=np.load(x_tmp, mmap_mode="r"), Y=np.load(y_tmp, mmap_mode="r"))

    x_tmp.unlink(missing_ok=True)
    y_tmp.unlink(missing_ok=True)

    print("Saved:", out_path)
    print("Shape X:", x_shape, "Shape Y:", (total_target,))
    print("Class counts:", counts.tolist())
    print("Total:", int(counts.sum()))


create_c5m_imbalanced(DIR_DATA)
