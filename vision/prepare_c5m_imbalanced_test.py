import numpy as np

DIR_DATA = "/project/home/p200535/data/cifar-5m-imbalanced/c5m_imbalanced.npz"
CLASS_COUNTS = np.array([500_000 // (i + 1) for i in range(10)], dtype=np.int64)

with np.load(DIR_DATA, mmap_mode="r") as data:
    y = data["Y"]
    counts = np.bincount(y, minlength=10)

np.testing.assert_array_equal(counts, CLASS_COUNTS)
assert int(counts.sum()) == int(CLASS_COUNTS.sum())

print("Class counts:", counts)
