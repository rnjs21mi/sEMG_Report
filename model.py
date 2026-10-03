"""모델 정의(제안 모델 + 베이스라인)와 학습 루프. 모든 모델을 같은 조건으로 학습한다."""
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, resnet18

BATCH, LR = 16, 1e-3
MODEL_NAMES = ["densenet161", "cnn2d", "resnet18"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, n_classes=5):
    if name == "densenet161":                      # 제안 모델, 사전학습 없음
        m = densenet161(weights=None)
        m.classifier = nn.Linear(2208, n_classes)
        return m
    if name == "resnet18":                         # 베이스라인 B
        m = resnet18(weights=None)
        m.fc = nn.Linear(512, n_classes)
        return m
    if name == "cnn2d":                            # 베이스라인 A: 간단한 2D CNN (강의 슬라이드 구조)
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, n_classes))
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _accuracy(model, loader, dev):
    model.eval(); correct = n = 0
    for xb, yb in loader:
        correct += (model(xb.to(dev)).argmax(1).cpu() == yb).sum().item(); n += len(yb)
    return correct / n


def train_model(model, Xtr, ytr, epochs, dev, Xte=None, yte=None,
                batch=BATCH, lr=LR, max_batches=None, verbose=True):
    """Adam · lr 1e-3 · batch 16 · CrossEntropy. 반환: (에폭별 로그, 총 학습 시간(초)).
    Xte가 주어지면 에폭마다 테스트 정확도를 '참고용'으로만 기록한다(모델 선택에는 쓰지 않는다)."""
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=batch, shuffle=True)
    te_loader = None
    if Xte is not None:
        te_loader = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=batch)
    model = model.to(dev)
    opt, crit = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    log, t_all = [], time.time()
    for ep in range(epochs):
        model.train(); tot = seen = 0; t0 = time.time()
        for bi, (xb, yb) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); seen += len(yb)
        row = {"epoch": ep + 1, "train_loss": tot / seen, "epoch_sec": time.time() - t0}
        if te_loader is not None:
            row["test_acc"] = _accuracy(model, te_loader, dev)
        log.append(row)
        if verbose:
            extra = f" | test acc {row['test_acc']*100:5.2f}%" if "test_acc" in row else ""
            print(f"  epoch {ep+1:>2}/{epochs} | loss {row['train_loss']:.4f}{extra} | {row['epoch_sec']:.0f}s", flush=True)
    return log, time.time() - t_all
