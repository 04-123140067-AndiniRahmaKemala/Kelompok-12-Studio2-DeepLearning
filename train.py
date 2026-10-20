"""Studio 2 - Klasifikasi Sampah: fine-tuning backbone pretrained (timm).

Contoh:
    python train.py --data ./train --submission ./submission --my-val ./my_val \
        --backbone efficientnet_b2 --img-size 288 --epochs 15

Hasil:
    submission/model.py, submission/model.pth, submission/requirements.txt
    my_val/               split validasi lokal (identik dengan train_baseline.py: 15%/kelas, seed 42)
    results/train_log.json

Uji dengan skrip resmi:
    python evaluate.py --submission ./submission --data ./my_val

Teknik: augmentasi kuat (RandomResizedCrop, TrivialAugmentWide, RandomErasing), label smoothing
(label noise), class weight (kelas tidak seimbang), AdamW + warmup 1 epoch + cosine decay,
mixed precision. Model terbaik dipilih berdasarkan Macro-F1 validasi lokal.
"""
import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

MODEL_PY = '''"""Submission Studio 2 - __BACKBONE__ (timm), input __IMG__ px."""
import timm
from torchvision import transforms

NUM_CLASSES = 5
IMG_SIZE = __IMG__
MEAN = __MEAN__
STD = __STD__


def get_model():
    """Bangun arsitektur saja - TIDAK mengunduh bobot / memuat model.pth di sini."""
    return timm.create_model("__BACKBONE__", pretrained=False, num_classes=NUM_CLASSES)


def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
'''


def build_parser():
    ap = argparse.ArgumentParser(description="Studio 2 - training klasifikasi sampah")
    ap.add_argument("--data", required=True, help="folder train/ (berisi 5 folder kelas)")
    ap.add_argument("--submission", default="./submission")
    ap.add_argument("--my-val", default="./my_val")
    ap.add_argument("--results", default="./results")
    ap.add_argument("--csv", default=None, help="train_sources.csv untuk cek kelengkapan data")
    ap.add_argument("--backbone", default="efficientnet_b2")
    ap.add_argument("--img-size", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--full-data", action="store_true",
                    help="latih pada SELURUH data (val lokal ikut dilatih; untuk putaran Final)")
    ap.add_argument("--no-pretrained", action="store_true")
    return ap


def check_dataset(src, csv_path):
    found = sorted(d.name for d in Path(src).iterdir() if d.is_dir())
    if found != CLASSES:
        raise SystemExit(f"[GAGAL] folder kelas tidak cocok: {found} (harusnya {CLASSES})")
    on_disk = {f"{c}/{p.name}" for c in CLASSES for p in (Path(src) / c).glob("*.jpg")}
    print(f"gambar di disk: {len(on_disk)}")
    if csv_path and Path(csv_path).exists():
        with open(csv_path, encoding="utf-8") as fh:
            expected = {r["file"] for r in csv.DictReader(fh)}
        missing = expected - on_disk
        print(f"train_sources.csv: {len(expected)} | hilang: {len(missing)}")
        if missing:
            per = {c: sum(m.startswith(c + "/") for m in missing) for c in CLASSES}
            print("PERINGATAN: dataset TIDAK lengkap, hilang per kelas:", per)
    return len(on_disk)


def make_split(src, frac, seed):
    """Identik dengan make_local_split() di train_baseline.py (tanpa menyalin file)."""
    rng = random.Random(seed)
    tr, va = [], []
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            (va if i < n_va else tr).append((str(f), CLASSES.index(cls_dir.name)))
    return tr, va


def macro_f1(y_true, y_pred, k=len(CLASSES)):
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    f1s = []
    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(k)) - tp
        fn = sum(cm[c]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(cm[c][c] for c in range(k)) / max(1, len(y_true))
    return sum(f1s) / k, acc, f1s, cm


class ImgDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, gts, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
        loss_sum += nn.functional.cross_entropy(out.float(), y).item() * x.size(0)
        n += x.size(0)
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    f1, acc, f1s, cm = macro_f1(gts, preds)
    return loss_sum / n, f1, acc, f1s, cm


def main():
    a = build_parser().parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device={device} | backbone={a.backbone} | img={a.img_size} | epochs={a.epochs} | full_data={a.full_data}")

    check_dataset(a.data, a.csv)
    tr_part, val_samples = make_split(a.data, a.val_frac, a.seed)
    train_samples = tr_part + val_samples if a.full_data else tr_part

    my_val = Path(a.my_val)
    if my_val.exists():
        shutil.rmtree(my_val)
    for path, y in val_samples:
        dst = my_val / CLASSES[y]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
    print(f"train: {len(train_samples)} | val lokal: {len(val_samples)} -> {my_val}")
    if a.full_data:
        print("PERINGATAN: --full-data aktif, val lokal ikut dilatih sehingga skornya tidak objektif.")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(a.img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([  # harus sama dengan get_transform() di submission/model.py
        transforms.Resize((a.img_size, a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    g = torch.Generator()
    g.manual_seed(a.seed)
    train_loader = DataLoader(ImgDataset(train_samples, train_tf), batch_size=a.batch_size, shuffle=True,
                              num_workers=a.workers, pin_memory=use_amp, generator=g, drop_last=True)
    val_loader = DataLoader(ImgDataset(val_samples, val_tf), batch_size=64, shuffle=False,
                            num_workers=a.workers, pin_memory=use_amp)

    model = timm.create_model(a.backbone, pretrained=not a.no_pretrained, num_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameter: {n_params / 1e6:.2f} M | perkiraan model.pth: {n_params * 4 / 1e6:.0f} MB")

    counts = np.bincount([y for _, y in train_samples], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32).to(device)
    print("class weights:", weights.cpu().numpy().round(2))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=a.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = a.epochs * steps_per_epoch
    warmup = steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + float(np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sub = Path(a.submission)
    sub.mkdir(parents=True, exist_ok=True)
    history, best_f1, t0 = [], -1.0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        val_loss, f1, acc, _, _ = evaluate(model, val_loader, device, use_amp)
        saved = ""
        if a.full_data or f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), sub / "model.pth")
            saved = " *saved*"
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "val_loss": val_loss,
                        "val_macro_f1": f1, "val_acc": acc})
        print(f"epoch {epoch:2d}/{a.epochs} | train_loss {run_loss / seen:.4f} | val_loss {val_loss:.4f} | "
              f"val Macro-F1 {f1:.4f} | acc {acc:.4f}{saved}")
    print(f"selesai {(time.time() - t0) / 60:.1f} menit")

    # tulis model.py + requirements.txt, lalu evaluasi ulang bobot yang tersimpan (fp32, seperti evaluate.py)
    (sub / "model.py").write_text(
        MODEL_PY.replace("__BACKBONE__", a.backbone).replace("__IMG__", str(a.img_size))
        .replace("__MEAN__", str(MEAN)).replace("__STD__", str(STD)), encoding="utf-8")
    (sub / "requirements.txt").write_text("timm\n", encoding="utf-8")

    model.load_state_dict(torch.load(sub / "model.pth", map_location=device, weights_only=True))
    _, f1, acc, f1s, cm = evaluate(model, val_loader, device, use_amp=False)
    print(f"\nMacro-F1 (val lokal, bobot tersimpan): {f1:.4f} | akurasi: {acc:.4f}")
    for c, v in zip(CLASSES, f1s):
        print(f"  F1 {c:<15} {v:.4f}")
    size_mb = (sub / "model.pth").stat().st_size / 1e6
    print(f"model.pth: {size_mb:.1f} MB" + (" -> > 100 MB: pakai Git LFS / GitHub Release" if size_mb > 100 else ""))

    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.json").write_text(json.dumps({
        "args": vars(a), "params_million": n_params / 1e6, "history": history,
        "final_val": {"macro_f1": f1, "accuracy": acc, "per_class_f1": dict(zip(CLASSES, f1s)),
                      "confusion_matrix": cm},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
