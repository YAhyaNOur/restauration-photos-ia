import random
from pathlib import Path

SEED = 42
TRAIN_HR = Path("data/raw/DIV2K_train_HR")
VALID_HR = Path("data/raw/DIV2K_valid_HR")
OUT = Path("splits")


def main():
    OUT.mkdir(exist_ok=True)
    names = sorted(f.name for f in TRAIN_HR.glob("*.png"))
    random.Random(SEED).shuffle(names)

    test = sorted(names[:100])
    train = sorted(names[100:])
    val = sorted(f.name for f in VALID_HR.glob("*.png"))

    (OUT / "train.txt").write_text("\n".join(train) + "\n", newline="\n")
    (OUT / "val.txt").write_text("\n".join(val) + "\n", newline="\n")
    (OUT / "test.txt").write_text("\n".join(test) + "\n", newline="\n")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()