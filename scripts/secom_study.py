"""The same calibration question on manufacturing sensor data (UCI SECOM).

The main study measures calibration under distribution shift on text. This
script asks whether the finding is specific to text or to LLMs by repeating it
on the setting where confidence gating is called *virtual metrology*: predict
pass/fail from process sensors, auto-release what the model is confident about,
send the rest to physical measurement.

The shift here is not constructed -- it is time. SECOM is 1,567 wafers from a
semiconductor line over three months, 590 sensor readings each. We train on the
early part of the period and ask what the confidences are worth later on.

    in-domain     : early period, out-of-fold predictions (K-fold cross-fitting)
    out-of-domain : later period, never seen in training

Every row gets exactly one prediction, from a model that did not train on it.
Later rows are partitioned across the fold models rather than scored by all of
them: scoring each row K times would inflate n and narrow the bootstrap
intervals without adding information.

Output is the JSONL format of `evaluate.py --dump-predictions`, so the analysis
is `calibration_study.py` unchanged. Logits are written already divided by the
fold's temperature (fitted in-domain only); run the study with --temperature 1.

Data: https://archive.ics.uci.edu/static/public/179/secom.zip -> data/secom/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

from jevlike.calibrate import fit_temperature, softmax_rows
from jevlike.metrics import bootstrap_ci, evaluate_probs

BRIER_WEIGHT = 0.5  # same loss as the main model: CE + 0.5 * Brier


def load_secom(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    """Features, labels (1 = fail) and timestamps, in chronological order."""
    x = np.genfromtxt(data_dir / "secom.data")
    y, t = [], []
    with (data_dir / "secom_labels.data").open(encoding="utf-8") as f:
        for line in f:
            label, stamp = line.strip().split(" ", 1)
            y.append(int(label == "1"))
            t.append(datetime.strptime(stamp.strip('"'), "%d/%m/%Y %H:%M:%S"))
    if len(y) != len(x):
        raise ValueError(f"{len(x)} feature rows but {len(y)} labels.")
    if any(b < a for a, b in zip(t, t[1:])):
        raise ValueError("Timestamps are not sorted; the chronological split would leak.")
    return x, np.asarray(y, dtype=np.int64), t


class Preprocessor:
    """Imputation and scaling fitted on training rows only.

    Statistics from the later period must not reach the model: that would hide
    exactly the drift being measured.
    """

    def __init__(self, x: np.ndarray, max_missing: float = 0.5, drop: bool = True) -> None:
        missing = np.isnan(x).mean(axis=0)
        self.median = np.nanmedian(np.where(np.isnan(x).all(axis=0), 0.0, x), axis=0)
        filled = np.where(np.isnan(x), self.median, x)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.keep = (missing <= max_missing) & (self.std > 0)
        if not drop:
            # Windowed input needs every time step to keep the same columns.
            # Uninformative ones stay in place and scale to a constant zero.
            self.std = np.where(self.std > 0, self.std, 1.0)
            self.keep = np.ones_like(self.keep)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(x), self.median, x)
        z = (filled[:, self.keep] - self.mean[self.keep]) / self.std[self.keep]
        # A sensor that drifts far outside its training range is the shift we
        # study, but unbounded z-scores would let one column decide the logit.
        return np.clip(z, -10.0, 10.0).astype(np.float32)


NEURAL_ARCHS = ("linear", "mlp", "mlp_deep", "resnet", "ft_transformer", "ensemble")
TREE_ARCHS = (
    "tree", "random_forest", "extra_trees", "adaboost", "gbdt", "xgboost", "lightgbm",
    "catboost", "isolation_forest",
)
CLASSIC_ARCHS = (
    "logreg_l2", "logreg_l1", "svm_rbf", "knn", "naive_bayes", "lda", "pca_logreg",
    "pls_da", "tabpfn", "tabpfn_ft",
)
ARCHS = NEURAL_ARCHS + TREE_ARCHS + CLASSIC_ARCHS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm1d(width), nn.Linear(width, width), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(width, width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.body(h)


class FTTransformer(nn.Module):
    """Each sensor becomes a token; a [CLS] token attends over all of them."""

    def __init__(self, d_in: int, dim: int = 16, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_in, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, dropout=0.2, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0])


def build_net(arch: str, d_in: int, hidden: int) -> nn.Module:
    if arch == "linear":
        return nn.Linear(d_in, 2)
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2)
        )
    if arch == "mlp_deep":
        return nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2),
        )
    if arch == "resnet":
        return nn.Sequential(
            nn.Linear(d_in, 128), ResBlock(128, 0.3), ResBlock(128, 0.3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2),
        )
    if arch == "ft_transformer":
        return FTTransformer(d_in)
    raise ValueError(f"Unknown architecture: {arch}")


class Ensemble:
    """Average the probabilities of independently initialised MLPs."""

    def __init__(self, members: list[nn.Module]) -> None:
        self.members = members

    def logits(self, x: np.ndarray) -> list[list[float]]:
        probs = np.mean(
            [softmax_rows(predict_logits(m, x), 1.0) for m in self.members], axis=0
        )
        return np.log(np.clip(probs, 1e-12, 1.0)).tolist()


def build_tree(arch: str, seed: int):
    """One reasonable setting per family; none of them is tuned."""
    if arch == "tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=seed)
    if arch == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    if arch == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    if arch == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier

        return AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed)
    if arch == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3, random_state=seed
        )
    if arch == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.5, random_state=seed, verbosity=0,
        )
    if arch == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=8, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.5, min_child_samples=10, random_state=seed,
            verbose=-1,
        )
    if arch == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=4, random_seed=seed, verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown tree architecture: {arch}")


class PLSDA:
    """PLS discriminant analysis, the chemometrics default for many correlated sensors.

    PLS regresses the 0/1 label on a few latent directions; a logistic link
    turns its score into a probability.
    """

    def __init__(self, n_components: int = 5) -> None:
        self.n_components = n_components

    def fit(self, x: np.ndarray, y: np.ndarray) -> PLSDA:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import LogisticRegression

        self.pls = PLSRegression(n_components=self.n_components).fit(x, y)
        self.link = LogisticRegression().fit(self.pls.predict(x).reshape(-1, 1), y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.link.predict_proba(self.pls.predict(x).reshape(-1, 1))


def build_classic(arch: str, seed: int):
    """Non-tree, non-neural baselines. One reasonable setting each, untuned."""
    from sklearn.linear_model import LogisticRegression

    if arch == "logreg_l2":
        return LogisticRegression(C=0.01, max_iter=5000)
    if arch == "logreg_l1":
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=0.05, max_iter=5000,
                                  random_state=seed)
    if arch == "svm_rbf":
        from sklearn.svm import SVC

        return SVC(C=1.0, gamma="scale", probability=True, random_state=seed)
    if arch == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=25, weights="distance")
    if arch == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()
    if arch == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if arch == "pca_logreg":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline

        return make_pipeline(PCA(n_components=20, random_state=seed),
                             LogisticRegression(max_iter=5000))
    if arch == "pls_da":
        return PLSDA()
    if arch == "tabpfn":
        # A transformer pretrained on synthetic tables; no training here, the
        # fold's rows are its context. Built for exactly this data size.
        # v2 weights: openly downloadable. Later versions need a Prior Labs
        # account, and the library opens a browser to get one unless told not to.
        import os

        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device=DEVICE, random_state=seed
        )
    if arch == "tabpfn_ft":
        # The same v2 weights, fine-tuned on the fold's training rows. The
        # library holds out 10% of them itself for early stopping.
        import os

        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        # Every wafer-sensor cell is a token, so a whole fold with gradients
        # does not fit 12 GB. Each step therefore sees 150 wafers, with one
        # estimator. The cap makes an overflow fail as a CUDA OOM; uncapped,
        # Windows spills it into system memory and the machine starts to swap.
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
        return FinetunedTabPFNClassifier(
            model_version=ModelVersion.V2, device=DEVICE, random_state=seed,
            n_finetune_ctx_plus_query_samples=150, n_estimators_finetune=1,
        )
    raise ValueError(f"Unknown classic architecture: {arch}")


class TreeModel:
    """Anything with fit / predict_proba: tree ensembles and classic baselines."""

    def __init__(self, arch: str, x: np.ndarray, y: np.ndarray, seed: int) -> None:
        build = build_tree if arch in TREE_ARCHS else build_classic
        self.model = build(arch, seed).fit(x, y)

    def logits(self, x: np.ndarray) -> list[list[float]]:
        # A leaf holding only passing wafers says P(fail) = 0. Floor it, or one
        # such wafer failing makes the log-loss infinite and wrecks temperature.
        probs = np.clip(self.model.predict_proba(x), 1e-4, 1.0)
        return np.log(probs).tolist()


class IsolationModel:
    """Anomaly detection: learns what a normal wafer looks like, not the labels.

    The forest never sees y. Labels are used only afterwards, to map its
    anomaly score onto a probability with a one-parameter logistic fit.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression

        self.forest = IsolationForest(n_estimators=300, random_state=seed).fit(x)
        self.link = LogisticRegression().fit(self._score(x), y)

    def _score(self, x: np.ndarray) -> np.ndarray:
        return -self.forest.score_samples(x).reshape(-1, 1)

    def logits(self, x: np.ndarray) -> list[list[float]]:
        probs = np.clip(self.link.predict_proba(self._score(x)), 1e-4, 1.0)
        return np.log(probs).tolist()


class Boosted(TreeModel):
    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int) -> None:
        super().__init__("gbdt", x, y, seed)


def top_sensors(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """The k columns whose mean differs most between failing and passing wafers."""
    gap = np.abs(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
    return np.sort(np.argsort(-gap)[:k])


class Subset:
    """Any model, restricted to a fixed set of sensors."""

    def __init__(self, model, cols: np.ndarray) -> None:
        self.model, self.cols = model, cols

    def logits(self, x: np.ndarray) -> list[list[float]]:
        return predict_logits(self.model, x[:, self.cols])


def train_mlp(
    x: np.ndarray, y: np.ndarray, *, seed: int, epochs: int, hidden: int, weight_decay: float,
    arch: str = "mlp", net_factory=None, n_classes: int = 2,
):
    """Fit one model. Every neural variant shares the loss and the optimiser.

    `net_factory(d_in)` supplies a network from outside this file; it is built
    after the seed is set, so its initialisation is reproducible too.
    """
    if arch == "isolation_forest":
        return IsolationModel(x, y, seed)
    if arch in TREE_ARCHS or arch in CLASSIC_ARCHS:
        return TreeModel(arch, x, y, seed)
    if arch == "ensemble":
        return Ensemble([
            train_mlp(x, y, seed=seed * 1000 + m, epochs=epochs, hidden=hidden,
                      weight_decay=weight_decay)
            for m in range(5)
        ])
    torch.manual_seed(seed)
    net = net_factory(x.shape[1]) if net_factory else build_net(arch, x.shape[1], hidden)
    model = net.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    xt, yt = torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)
    onehot = nn.functional.one_hot(yt, n_classes).float()
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), generator=gen).split(64):
            if len(idx) < 2:  # BatchNorm cannot normalise a single row
                continue
            idx = idx.to(DEVICE)
            logits = model(xt[idx])
            ce = nn.functional.cross_entropy(logits, yt[idx])
            brier = ((logits.softmax(-1) - onehot[idx]) ** 2).sum(-1).mean()
            loss = ce + BRIER_WEIGHT * brier
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict_logits(model, x: np.ndarray) -> list[list[float]]:
    if hasattr(model, "logits"):
        return model.logits(x)
    xt = torch.from_numpy(x).to(DEVICE)
    return torch.cat([model(chunk) for chunk in xt.split(256)]).double().cpu().tolist()


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability that a failing wafer is scored above a passing one."""
    ranks = rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def auc_ci(score: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(auc(score[idx], y[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def cross_fit(
    x: np.ndarray, y: np.ndarray, t: list[datetime], a, *, verbose: bool = False, trainer=None
):
    """One prediction per row, each from a model that did not train on it.

    `trainer(x_train, y_train, seed)` replaces the built-in architectures; it
    must return something `predict_logits` accepts.
    """
    n_early = int(len(y) * a.split_frac)
    rng = np.random.default_rng(a.seed)
    if a.random_split:
        order = rng.permutation(len(y))
        x, y, t = x[order], y[order], [t[i] for i in order]
    early_fold = rng.permutation(n_early) % a.folds
    late_fold = rng.permutation(len(y) - n_early) % a.folds

    rows: list[dict] = []
    temps: list[float] = []
    for k in range(a.folds):
        fit_idx = np.flatnonzero(early_fold != k)
        rng.shuffle(fit_idx)
        n_cal = int(len(fit_idx) * a.cal_frac)
        cal_idx, train_idx = fit_idx[:n_cal], fit_idx[n_cal:]
        test_idx = np.flatnonzero(early_fold == k)
        late_idx = n_early + np.flatnonzero(late_fold == k)

        prep = Preprocessor(x[train_idx], drop=getattr(a, "drop_columns", True))
        if trainer is not None:
            model = trainer(prep(x[train_idx]), y[train_idx], a.seed + k)
        else:
            # Sensors are chosen per fold from its training rows; choosing them
            # on all the data would leak the later period into the model.
            xt = prep(x[train_idx])
            cols = top_sensors(xt, y[train_idx], a.top_k) if a.top_k else np.arange(xt.shape[1])
            model = Subset(
                train_mlp(
                    xt[:, cols], y[train_idx], seed=a.seed + k, epochs=a.epochs,
                    hidden=a.hidden, weight_decay=a.weight_decay, arch=a.arch,
                ),
                cols,
            )
        cal = fit_temperature(predict_logits(model, prep(x[cal_idx])), y[cal_idx].tolist())
        temps.append(cal.temperature)
        if verbose:
            print(f"fold {k}: train={len(train_idx)} cal={len(cal_idx)}  {cal.summary()}")

        for name, idx in (("in_task", test_idx), ("held_out", late_idx)):
            for i, raw in zip(idx, predict_logits(model, prep(x[idx]))):
                rows.append(
                    {
                        "set": name,
                        "source": t[i].strftime("%Y-%m"),
                        "time": t[i].isoformat(),
                        "fold": k,
                        "label": int(y[i]),
                        "raw_logits": raw,
                        "logits": [v / cal.temperature for v in raw],
                    }
                )
    return rows, temps, n_early, y, t


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration under temporal shift on SECOM")
    ap.add_argument("--data-dir", type=Path, default=Path("data/secom"))
    ap.add_argument("--out", type=Path, default=Path("runs/secom/preds.jsonl"))
    ap.add_argument("--split-frac", type=float, default=0.6,
                    help="fraction of the period (by row order) treated as in-domain")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.2,
                    help="share of each fold's training rows held back to fit temperature")
    ap.add_argument("--arch", choices=ARCHS, default="mlp")
    ap.add_argument("--top-k", type=int, default=0,
                    help="keep only the k most informative sensors (0 = all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--release-at", type=float, default=0.95)
    ap.add_argument("--random-split", action="store_true",
                    help="control: shuffle rows before splitting, so 'held_out' is not later in "
                         "time. Any gap that survives this is not caused by temporal shift.")
    a = ap.parse_args()

    x, y, t = load_secom(a.data_dir)
    rows, temps, n_early, y, t = cross_fit(x, y, t, a, verbose=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(
        f"\nin-domain  {t[0]:%Y-%m-%d} .. {t[n_early - 1]:%Y-%m-%d}  "
        f"n={n_early}  fail rate={y[:n_early].mean():.3f}"
    )
    print(
        f"out-of-dom {t[n_early]:%Y-%m-%d} .. {t[-1]:%Y-%m-%d}  "
        f"n={len(y) - n_early}  fail rate={y[n_early:].mean():.3f}"
    )
    print(f"temperature per fold: {', '.join(f'{v:.3f}' for v in temps)}\n")
    for name in ("in_task", "held_out"):
        rs = [r for r in rows if r["set"] == name]
        labels = [r["label"] for r in rs]
        for key in ("raw_logits", "logits"):
            probs = softmax_rows([r[key] for r in rs], 1.0)
            ci = bootstrap_ci(probs, labels, "ece", n_boot=a.n_boot)
            tag = "T=1    " if key == "raw_logits" else "T=fold "
            print(
                f"{name:9s} {tag} {evaluate_probs(probs, labels).summary()}  "
                f"ECE 95% [{ci['lo']:.4f}, {ci['hi']:.4f}]"
            )

    # Accuracy and ECE follow the base rate, which falls over the period, so
    # they can improve while the model stops telling good wafers from bad.
    # Ranking and the release gate are what the factory actually relies on.
    print(f"\nrelease gate: auto-release when P(pass) >= {a.release_at}")
    for name in ("in_task", "held_out"):
        rs = [r for r in rows if r["set"] == name]
        labels = np.asarray([r["label"] for r in rs])
        p_fail = np.asarray(softmax_rows([r["logits"] for r in rs], 1.0))[:, 1]
        lo, hi = auc_ci(p_fail, labels, n_boot=a.n_boot, seed=a.seed)
        rel = (1.0 - p_fail) >= a.release_at
        print(
            f"{name:9s} AUC={auc(p_fail, labels):.3f} [{lo:.3f}, {hi:.3f}]  "
            f"released {int(rel.sum())}/{len(rel)}: fail rate {labels[rel].mean():.4f}  "
            f"held back: fail rate {labels[~rel].mean():.4f}"
        )
    print(f"\n-> {a.out}")
    print(f"next: python scripts/calibration_study.py --predictions {a.out} --temperature 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
