"""
train.py - build features, train decision tree (FPGA) vs baselines, split by recording.

  python train.py                 # synthetic data
Swap in real data: replace make_dataset() with a loader returning the same dicts
(cls, rec_id, x=int16 @256 kHz, intervals=[] -> then label events by recording class).
"""
import json, pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from synth_bats import make_dataset, CLASSES, RISK
from bat_model import extract, FEATURES

MAX_DEPTH = 8


def risk(c):
    return RISK.get(c, "other_bat")


def main():
    recs = make_dataset(n_rec_per_class=40, seed=0)
    rows = [ev for r in recs for ev in extract(r)]
    X = np.array([r["feats"] for r in rows])
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    rec_idx = np.array([int(r["rec_id"].split("_")[1]) for r in rows])
    test = rec_idx >= 30                       # last 10 recordings/class held out: no individual leakage
    print(f"{len(rows)} detected calls  train={(~test).sum()} test={test.sum()}")
    print("calls per class:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})

    models = {
        f"decision_tree_d{MAX_DEPTH} (FPGA)": DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=3, random_state=0),
        "random_forest_100": RandomForestClassifier(100, random_state=0),
        "grad_boost": HistGradientBoostingClassifier(random_state=0),
    }
    results = {}
    for name, m in models.items():
        m.fit(X[~test], y[~test])
        p = m.predict(X[test])
        acc = accuracy_score(y[test], p)
        r_true = [risk(CLASSES[i]) for i in y[test]]; r_pred = [risk(CLASSES[i]) for i in p]
        racc = np.mean(np.array(r_true) == np.array(r_pred))
        results[name] = dict(species_acc=round(acc, 3), risk_group_acc=round(racc, 3))
        print(f"{name:32s} species acc {acc:.3f}   risk-group acc {racc:.3f}")

    dt = models[f"decision_tree_d{MAX_DEPTH} (FPGA)"]
    p = dt.predict(X[test])
    labels = list(range(len(CLASSES)))
    print(classification_report(y[test], p, labels=labels, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y[test], p, labels=labels)
    np.savetxt("confusion_matrix.csv", cm, fmt="%d", delimiter=",", header=",".join(CLASSES))
    print(f"tree: {dt.get_n_leaves()} leaves, depth {dt.get_depth()}")

    pickle.dump(dict(tree=dt, classes=CLASSES, features=FEATURES), open("tree.pkl", "wb"))
    json.dump(dict(results=results, classes=CLASSES, risk={c: risk(c) for c in CLASSES}),
              open("results.json", "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(labels, CLASSES, rotation=90); ax.set_yticks(labels, CLASSES)
        for i in labels:
            for j in labels:
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Decision tree (depth {MAX_DEPTH}) on held-out recordings - SYNTHETIC data")
        fig.tight_layout(); fig.savefig("confusion_matrix.png", dpi=130)
    except ImportError:
        pass
    return recs


if __name__ == "__main__":
    main()
