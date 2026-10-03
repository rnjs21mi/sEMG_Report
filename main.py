"""전체 실험 실행: 데이터 준비 → 모델별 학습·평가 → (선택) 5-fold 교차검증.

예)  python main.py --data-dir palm-sEMG-doorknob-filtered/data --epochs 45 --seeds 42 43 44 --cv
빠른 점검: python main.py --epochs 1 --max-batches 3 --models cnn2d --infer-iters 5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import list_trials, load_filtered, split_trials, build, CLASSES
import evaluation as ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="palm-sEMG-doorknob-filtered/data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["densenet161", "cnn2d", "resnet18"])
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="초기화·셔플 시드(분할은 고정). 3개면 3회 평균")
    ap.add_argument("--cv", action="store_true", help="5-fold 교차검증 실행")
    ap.add_argument("--cv-model", default="densenet161")
    ap.add_argument("--cv-epochs", type=int, default=None, help="기본값: --epochs와 동일")
    ap.add_argument("--max-batches", type=int, default=None, help="동작 확인용")
    ap.add_argument("--infer-iters", type=int, default=100)
    args = ap.parse_args()

    import torch
    from model import build_model, train_model, count_params, set_seed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(exist_ok=True)
    print("device:", dev)

    files, labels = list_trials(args.data_dir)
    print("시행 수:", len(files), dict(zip(*np.unique(labels, return_counts=True))))
    filt = load_filtered(files)

    tr_files, te_files = split_trials(files, labels)
    t0 = time.time()
    Xtr, ytr = build(tr_files, filt); Xte, yte = build(te_files, filt)
    print(f"학습 시행 {len(tr_files)} / 테스트 시행 {len(te_files)} (겹침 0) | Xtr {Xtr.shape} Xte {Xte.shape} | CWT {time.time()-t0:.0f}s")
    pd.DataFrame({"file": [f.name for f in tr_files + te_files],
                  "subject": [f.parent.name for f in tr_files + te_files],
                  "split": ["train"] * len(tr_files) + ["test"] * len(te_files)}).to_csv(out / "split_trials.csv", index=False)

    summary, per_seed = [], []
    for name in args.models:
        runs = []
        for si, seed in enumerate(args.seeds):
            print(f"\n=== {name} | seed {seed} ({si+1}/{len(args.seeds)}) ===")
            set_seed(seed)
            m = build_model(name)
            log, sec = train_model(m, Xtr, ytr, args.epochs, dev, Xte, yte, max_batches=args.max_batches)
            pred = ev.predict(m, Xte, dev)
            met = ev.compute_metrics(yte, pred)
            cm = ev.get_confusion(yte, pred)
            runs.append({"model": name, "seed": seed, **met, "train_sec": sec})
            print(f"  test acc {met['accuracy']*100:.2f}% | macro F1 {met['f1']*100:.2f}%")
            if si == 0:                                    # 첫 시드: 혼동행렬·비용·로그 저장
                print(ev.report(yte, pred)); print(cm)
                ev.save_confusion(cm, name, out / f"confusion_{name}.png")
                np.savetxt(out / f"confusion_{name}.csv", cm, fmt="%d", delimiter=",")
                pd.DataFrame(log).to_csv(out / f"train_log_{name}.csv", index=False)
                inf_ms = ev.measure_inference_ms(m, Xte, dev, iters=args.infer_iters)
                a, b, n = ev.max_error_pair(cm)
                info = {"params": count_params(m), "inference_ms": inf_ms, "max_error": f"{a}->{b} ({n})"}
                torch.save(m.state_dict(), out / f"{name}.pt")
        per_seed += runs
        df = pd.DataFrame(runs)
        row = {"model": name, "n_runs": len(runs)}
        for k in ["accuracy", "precision", "recall", "f1"]:
            row[k] = df[k].mean(); row[k + "_std"] = df[k].std(ddof=0)
        row.update(train_sec=df.train_sec.mean(), **info)
        summary.append(row)
        pd.DataFrame(per_seed).to_csv(out / "results_per_seed.csv", index=False)
        pd.DataFrame(summary).to_csv(out / "results_summary.csv", index=False)

    # README에 붙여 넣을 표
    lines = ["| Model | Accuracy | Precision | Recall | F1-score | 학습 시간(초) | 파라미터 수 | 추론(ms/샘플) | 최대 오류 쌍 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['model']} | {r['accuracy']*100:.2f} | {r['precision']*100:.2f} | {r['recall']*100:.2f} | "
                     f"{r['f1']*100:.2f} | {r['train_sec']:.0f} | {r['params']:,} | {r['inference_ms']:.2f} | {r['max_error']} |")
    (out / "results_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    del Xtr, ytr, Xte, yte
    if args.cv:
        ep = args.cv_epochs or args.epochs
        print(f"\n=== 5-fold CV | {args.cv_model} | {ep} epochs ===")
        cv = ev.cross_validate(files, labels, filt, args.cv_model, ep, args.seeds[0], dev,
                               out / f"cv_{args.cv_model}.csv", max_batches=args.max_batches)
        a = cv.accuracy.values * 100
        txt = (f"5-fold accuracy: {a.round(2).tolist()}\n평균 {a.mean():.2f} ± {a.std(ddof=0):.2f} % (ddof=0) "
               f"/ ± {a.std(ddof=1):.2f} % (ddof=1)")
        (out / "cv_summary.txt").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
