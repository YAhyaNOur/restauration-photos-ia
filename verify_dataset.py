
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent

FOLDERS = {
    "HR train": ROOT / "data/raw/DIV2K_train_HR",
    "HR validation": ROOT / "data/raw/DIV2K_valid_HR",
    "LR train": ROOT / "data/processed/train_LR",
    "LR validation": ROOT / "data/processed/valid_LR",
}

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images_in(folder):
    if not folder.exists():
        return {}
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }


print("=" * 55)
print("1. NOMBRE D'IMAGES")
print("=" * 55)

datasets = {}

for name, folder in FOLDERS.items():
    datasets[name] = images_in(folder)
    print(f"{name}: {len(datasets[name])} images")

print("\n" + "=" * 55)
print("2. CORRESPONDANCE HR / LR")
print("=" * 55)

pairs = [
    ("HR train", "LR train"),
    ("HR validation", "LR validation"),
]

for hr_name, lr_name in pairs:
    hr = datasets[hr_name]
    lr = datasets[lr_name]

    missing_lr = sorted(set(hr) - set(lr))
    missing_hr = sorted(set(lr) - set(hr))

    print(f"\n{hr_name} ↔ {lr_name}")
    print(f"Correspondances: {len(set(hr) & set(lr))}")
    print(f"LR manquantes: {len(missing_lr)}")
    print(f"HR manquantes: {len(missing_hr)}")

print("\n" + "=" * 55)
print("3. DIMENSIONS ET FACTEUR ×4")
print("=" * 55)

for hr_name, lr_name in pairs:
    hr = datasets[hr_name]
    lr = datasets[lr_name]
    checked = 0
    errors = []

    for stem in sorted(set(hr) & set(lr)):
        try:
            with Image.open(hr[stem]) as im_hr:
                w_hr, h_hr = im_hr.size
            with Image.open(lr[stem]) as im_lr:
                w_lr, h_lr = im_lr.size

            checked += 1
            if w_hr != 4 * w_lr or h_hr != 4 * h_lr:
                errors.append(
                    f"{stem}: HR={w_hr}x{h_hr}, LR={w_lr}x{h_lr}"
                )
        except Exception as exc:
            errors.append(f"{stem}: erreur de lecture: {exc}")

    print(f"\n{hr_name}: {lr_name}")
    print(f"Paires vérifiées: {checked}")
    print(f"Dimensions non conformes au ×4: {len(errors)}")

    for error in errors[:5]:
        print("  -", error)

print("\n" + "=" * 55)
print("4. FICHIERS DE SPLIT")
print("=" * 55)

split_dir = ROOT / "splits"
split_names = ["train.txt", "val.txt", "test.txt"]
splits = {}

for filename in split_names:
    path = split_dir / filename

    if not path.exists():
        print(f"{filename}: INTROUVABLE")
        continue

    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    # Compare les noms sans extension pour détecter les doublons.
    ids = [Path(entry.replace("\\", "/")).stem for entry in entries]
    splits[filename] = set(ids)

    print(
        f"{filename}: {len(entries)} lignes, "
        f"{len(set(ids))} identifiants uniques, "
        f"{len(ids) - len(set(ids))} doublon(s)"
    )

for i, first in enumerate(split_names):
    if first not in splits:
        continue

    for second in split_names[i + 1:]:
        if second not in splits:
            continue

        overlap = splits[first] & splits[second]
        print(f"Chevauchement {first} / {second}: {len(overlap)}")
        if overlap:
            print("  Exemples:", sorted(overlap)[:5])

print("\nContrôle terminé. Aucune donnée n'a été modifiée.")
