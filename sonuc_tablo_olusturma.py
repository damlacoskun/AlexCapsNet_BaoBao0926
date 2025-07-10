import os
import re
from pathlib import Path
import pandas as pd
import math

def parse_eval_file(filepath: Path):
    """
    eval_result.txt içindeki tüm epoch'ları okuyup,
    en iyi accuracy (max), en düşük loss (min), en iyi precision (max),
    en iyi recall (max) ve en iyi F1 (max) değerlerini ve epoch numaralarını döner.
    """
    best = {
        'best_accuracy': -math.inf, 'best_accuracy_epoch': None,
        'best_loss':     math.inf,  'best_loss_epoch':     None,
        'best_precision':-math.inf, 'best_precision_epoch':None,
        'best_recall':   -math.inf, 'best_recall_epoch':   None,
        'best_F1':       -math.inf, 'best_F1_epoch':       None,
    }

    text = filepath.read_text()
    # Epoch bloklarını ayırıyoruz (çift boş satır bazlı)
    blocks = [b for b in text.split('\n\n') if b.strip()]
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 6:
            continue  # eksik blokları atla
        try:
            epoch = int(re.search(r'the epoch:\s*(\d+)',      lines[0], re.IGNORECASE).group(1))
            acc   = float(re.search(r'The accuracy:\s*([0-9]*\.?[0-9]+)', lines[1]).group(1))
            loss  = float(re.search(r'The loss:\s*([0-9]*\.?[0-9]+)',     lines[2]).group(1))
            prec  = float(re.search(r'The precision:\s*([0-9]*\.?[0-9]+)',lines[3]).group(1))
            rec   = float(re.search(r'The recall:\s*([0-9]*\.?[0-9]+)',   lines[4]).group(1))
            f1    = float(re.search(r'The F1:\s*([0-9]*\.?[0-9]+)',       lines[5]).group(1))
        except Exception:
            continue

        # Güncellemeler
        if acc   > best['best_accuracy']:
            best['best_accuracy'] = acc
            best['best_accuracy_epoch'] = epoch
        if loss  < best['best_loss']:
            best['best_loss'] = loss
            best['best_loss_epoch'] = epoch
        if prec  > best['best_precision']:
            best['best_precision'] = prec
            best['best_precision_epoch'] = epoch
        if rec   > best['best_recall']:
            best['best_recall'] = rec
            best['best_recall_epoch'] = epoch
        if f1    > best['best_F1']:
            best['best_F1'] = f1
            best['best_F1_epoch'] = epoch

    return best

def main():
    # Tüm seed klasörlerinin bulunduğu dizin (Path objesi)
    BASE_DIR = Path("./Result/BreastMNIST_SE/5.AlexCapsNet_Recon/train")
    rows = []

    # Her bir alt klasörü dolaşıyoruz
    for seed_dir in sorted(BASE_DIR.iterdir(), key=lambda p: p.name):
        if not seed_dir.is_dir():
            continue
        eval_file = seed_dir / 'eval_result.txt'
        if eval_file.exists():
            best = parse_eval_file(eval_file)
            best['seed'] = seed_dir.name
            rows.append(best)

    # DataFrame oluştur ve kolonları istediğimiz sıraya göre düzenle
    df = pd.DataFrame(rows)
    cols = [
        'seed',
        'best_accuracy',       'best_accuracy_epoch',
        'best_loss',           'best_loss_epoch',
        'best_precision',      'best_precision_epoch',
        'best_recall',         'best_recall_epoch',
        'best_F1',             'best_F1_epoch'
    ]
    df = df[cols]

    # Excel olarak kaydet
    out_path = BASE_DIR / 'Seeds_eval_summary.xlsx'
    df.to_excel(out_path, index=False)
    print(f"Özet tablo kaydedildi: {out_path}")

if __name__ == '__main__':
    main()
