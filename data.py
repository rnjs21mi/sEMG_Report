"""데이터 로딩 · 시행 단위 분할 · 필터 · 윈도우 · 정규화 · CWT (2주차 노트북과 동일한 처리)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.model_selection import train_test_split

FS = 1000                    # 샘플링 주파수 (Hz)
WIN, HOP = 300, 150          # 윈도우 300 ms, 홉 150 ms (50% 오버랩)
SCALES = np.arange(1, 33)    # CWT 스케일 32개
N_WIN = (3000 - WIN) // HOP + 1   # 시행당 윈도우 19개
CLASSES = list("ABCDE")
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
SPLIT_SEED = 42

_natural = lambda p: int(re.search(r"\((\d+)\)", p.name).group(1))   # 'a (12).csv' -> 12


def list_trials(data_dir):
    """폴더명(A~E)을 라벨로 사용. 반환: (파일 경로 리스트, 라벨 배열)."""
    files, labels = [], []
    for cls in CLASSES:
        for p in sorted((Path(data_dir) / cls).glob("*.csv"), key=_natural):
            files.append(p)
            labels.append(cls)
    return files, np.array(labels)


def preprocess(x, fs=FS):
    """60 Hz 노치 → 20~499 Hz 4차 대역통과 (filtfilt, axis=0). x: (시간, 채널)."""
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20 / (fs / 2), 499 / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def load_filtered(files):
    """모든 시행을 읽어 필터를 적용한 딕셔너리 {경로: (3000, 2)}."""
    out = {}
    for p in files:
        x = pd.read_csv(p).values.astype(np.float64)
        assert x.shape == (3000, 2), (p, x.shape)
        out[p] = preprocess(x)
    return out


def make_windows(x, win=WIN, hop=HOP):
    """(시간, 채널) -> (윈도우수, win, 채널)"""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i * hop: i * hop + win] for i in range(n)])


def minmax(w, eps=1e-8):
    """윈도우 단위 min-max (두 채널을 합쳐서 정규화: 2주차 기본 설정)."""
    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)
    return (w - mn) / (mx - mn + eps)


def to_cwt(one_window, wavelet="morl"):
    """(300, 2) -> (3, 32, 300): 채널1, 채널2, 두 채널 평균의 |CWT|."""
    maps = []
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        maps.append(np.abs(coef))
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)


def split_trials(files, labels, seed=SPLIT_SEED):
    """시행(파일) 단위 8:2 계층 분할. 윈도우는 분할 이후에 만든다(누수 방지)."""
    tr, te = train_test_split(files, test_size=0.2, stratify=labels, random_state=seed)
    assert not set(tr) & set(te), "같은 시행이 학습/테스트에 모두 있음"
    return tr, te


def build(flist, filt):
    """시행 리스트 -> (X: (N,3,32,300) float32, y: (N,) int64). 각 시행에서 윈도우 19개."""
    X = np.empty((len(flist) * N_WIN, 3, len(SCALES), WIN), np.float32)
    y = np.empty(len(flist) * N_WIN, np.int64)
    k = 0
    for f in flist:
        for w in minmax(make_windows(filt[f])):
            X[k] = to_cwt(w)
            y[k] = CLS2IDX[f.parent.name]
            k += 1
    return X, y
