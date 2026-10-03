"""성능 평가: 지표, 혼동행렬, 최대 오류 쌍, 계산 비용, 5-fold 교차검증."""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold

from data import CLASSES, build


def predict(model, X, dev, batch=64):
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0)


def get_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))


def max_error_pair(cm):
    """가장 큰 비대각 칸. 반환: (실제 클래스, 예측 클래스, 건수). 원본 cm은 바꾸지 않는다."""
    c = cm.copy(); np.fill_diagonal(c, 0)
    i, j = np.unravel_index(c.argmax(), c.shape)
    return CLASSES[i], CLASSES[j], int(c[i, j])


def save_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    a, b, n = max_error_pair(cm)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{title}\nmax error: {a}->{b} ({n})")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def measure_inference_ms(model, sample, dev, warmup=10, iters=100):
    """샘플 1개당 추론 시간(ms). 예열 후 측정하고 GPU면 synchronize."""
    import torch
    model.eval(); x = torch.from_numpy(sample[:1]).to(dev)
    sync = torch.cuda.synchronize if str(dev).startswith("cuda") else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(); t0 = time.time()
        for _ in range(iters):
            model(x)
        sync()
    return (time.time() - t0) / iters * 1000


def cross_validate(files, labels, filt, model_name, epochs, seed, dev, out_csv,
                   n_splits=5, max_batches=None):
    """시행(파일) 단위 StratifiedKFold. 폴드마다 모델을 새로 만들고, 끝난 폴드는 CSV에 즉시 저장
    (Colab 연결이 끊겨도 이어서 실행 가능)."""
    import torch
    from model import build_model, train_model, set_seed
    out_csv = Path(out_csv)
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {int(r["fold"]) for r in rows}
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    for k, (tr_i, va_i) in enumerate(skf.split(files, labels), start=1):
        if k in done:
            print(f"[CV] fold {k} 건너뜀(이미 완료)"); continue
        Xtr, ytr = build([files[i] for i in tr_i], filt)
        Xva, yva = build([files[i] for i in va_i], filt)
        set_seed(seed + k)
        m = build_model(model_name)                       # 폴드마다 새 모델
        _, sec = train_model(m, Xtr, ytr, epochs, dev, max_batches=max_batches, verbose=False)
        met = compute_metrics(yva, predict(m, Xva, dev))
        rows.append({"fold": k, **met, "train_sec": sec})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[CV] fold {k}: acc {met['accuracy']*100:.2f}% f1 {met['f1']*100:.2f}% ({sec:.0f}s)", flush=True)
        del Xtr, ytr, Xva, yva, m; gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
    return pd.DataFrame(rows).sort_values("fold")
