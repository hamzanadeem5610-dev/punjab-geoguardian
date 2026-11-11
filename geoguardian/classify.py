"""Land classification (PDF section 13).

Option A - rule-based NDVI/NDBI thresholds (baseline). The starting thresholds
           are tuned by grid search on the training samples.
Option B - machine learning: Random Forest and XGBoost on
           [Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDBI, MNDWI].

The model is evaluated with *spatial block cross-validation* (GroupKFold over
the sampling blocks). A random split would put neighbouring, near-identical
pixels into both train and test folds and overstate accuracy. The model with the
best cross-validated macro-F1 maps both years. One model trained on pooled samples
from all years keeps the class definitions identical across years, which matters
for change detection.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from .config import BUILT, FEATURES, OTHER, VEG, ProjectConfig
from .indices import YearData

NODATA = 255


# ------------------------------------------------------------------ Option A
@dataclass
class RuleThresholds:
    ndvi_veg: float
    ndbi_built: float
    mndwi_water: float


def rule_classify(ndvi, ndbi, mndwi, th: RuleThresholds) -> np.ndarray:
    ndvi, ndbi, mndwi = (np.asarray(a, np.float32) for a in (ndvi, ndbi, mndwi))
    out = np.full(ndvi.shape, OTHER, np.uint8)
    out[(ndbi >= th.ndbi_built) & (ndvi < th.ndvi_veg)] = BUILT
    out[ndvi >= th.ndvi_veg] = VEG
    out[(mndwi >= th.mndwi_water) & (ndvi < th.ndvi_veg)] = OTHER  # water / wet soil
    out[np.isnan(ndvi) | np.isnan(ndbi)] = NODATA
    return out


def tune_rule_thresholds(train: pd.DataFrame, cfg: ProjectConfig) -> Dict:
    start = RuleThresholds(cfg.rule_ndvi_veg, cfg.rule_ndbi_built, cfg.rule_mndwi_water)
    y = train["class_id"].values

    def score(th):
        p = rule_classify(train["NDVI"], train["NDBI"], train["MNDWI"], th)
        return f1_score(y, p, average="macro")

    best, best_s = start, score(start)
    for v, b, w in itertools.product(np.arange(0.20, 0.56, 0.025), np.arange(-0.20, 0.151, 0.025),
                                     np.arange(-0.10, 0.31, 0.05)):
        th = RuleThresholds(round(float(v), 3), round(float(b), 3), round(float(w), 3))
        s = score(th)
        if s > best_s + 1e-9:
            best, best_s = th, s
    return {"initial": start.__dict__, "initial_macro_f1": round(score(start), 4),
            "tuned": best.__dict__, "tuned_macro_f1": round(best_s, 4)}


# ------------------------------------------------------------------ Option B
def make_model(name: str, seed: int):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, objective="multi:softprob",
                             num_class=3, eval_metric="mlogloss", tree_method="hist",
                             n_jobs=-1, random_state=seed)
    raise ValueError(name)


@dataclass
class TrainedModel:
    name: str
    model: object
    cv: Dict
    all_cv: Dict[str, Dict] = field(default_factory=dict)
    importances: Optional[pd.Series] = None
    rule: Optional[Dict] = None


def _metrics(y, p) -> Dict:
    return {"overall_accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "kappa": round(float(cohen_kappa_score(y, p)), 4),
            "confusion_matrix": confusion_matrix(y, p, labels=[VEG, BUILT, OTHER]).tolist()}


def train_and_select(train: pd.DataFrame, cfg: ProjectConfig) -> TrainedModel:
    X = train[FEATURES].values.astype(np.float32)
    y = train["class_id"].values.astype(int)
    groups = train["block"].values
    n_splits = min(cfg.cv_folds, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    results: Dict[str, Dict] = {}

    rule_info = tune_rule_thresholds(train, cfg)
    th = RuleThresholds(**rule_info["tuned"])
    results["rule_based"] = _metrics(y, rule_classify(train["NDVI"], train["NDBI"],
                                                      train["MNDWI"], th))
    results["rule_based"]["note"] = "thresholds tuned on the same samples (optimistic)"

    for name in cfg.models:
        try:
            make_model(name, cfg.random_seed)
        except ImportError:
            print(f"[classify] {name} not installed - skipped")
            continue
        pred = np.zeros_like(y)
        for tr, te in cv.split(X, y, groups):
            m = make_model(name, cfg.random_seed)
            m.fit(X[tr], y[tr])
            pred[te] = np.asarray(m.predict(X[te])).astype(int).ravel()
        results[name] = _metrics(y, pred)
        print(f"[classify] {name:14s} spatial-CV  OA={results[name]['overall_accuracy']:.3f}  "
              f"macro-F1={results[name]['macro_f1']:.3f}  kappa={results[name]['kappa']:.3f}")
    ml = {k: v for k, v in results.items() if k != "rule_based"}
    if not ml:
        raise RuntimeError("No ML model could be trained")
    best = max(ml, key=lambda k: ml[k]["macro_f1"])
    model = make_model(best, cfg.random_seed)
    model.fit(X, y)
    imp = pd.Series(getattr(model, "feature_importances_", np.zeros(len(FEATURES))),
                    index=FEATURES).sort_values(ascending=False)
    print(f"[classify] selected model: {best}")
    return TrainedModel(name=best, model=model, cv=results[best], all_cv=results,
                        importances=imp, rule=rule_info)


# ----------------------------------------------------------------- mapping
def predict_year(tm: TrainedModel, yd: YearData, chunk: int = 500_000):
    """Class map (uint8, 255 = no data) and confidence (max class probability)."""
    shape = yd.valid.shape
    cls = np.full(shape, NODATA, np.uint8)
    conf = np.full(shape, np.nan, np.float32)
    idx = np.flatnonzero(yd.valid)
    X = yd.feature_stack().astype(np.float32)
    ok = ~np.isnan(X).any(axis=1)
    idx, X = idx[ok], X[ok]
    preds, confs = [], []
    for s in range(0, len(X), chunk):
        proba = tm.model.predict_proba(X[s:s + chunk])
        preds.append(np.argmax(proba, axis=1).astype(np.uint8))
        confs.append(proba.max(axis=1).astype(np.float32))
    if preds:
        cls.flat[idx] = np.concatenate(preds)
        conf.flat[idx] = np.concatenate(confs)
    return cls, conf


def rule_map(yd: YearData, rule_info: Dict) -> np.ndarray:
    th = RuleThresholds(**rule_info["tuned"])
    out = rule_classify(yd.idx["NDVI"], yd.idx["NDBI"], yd.idx["MNDWI"], th)
    out[~yd.valid] = NODATA
    return out


def majority_filter(cls: np.ndarray, size: int = 3) -> np.ndarray:
    """3x3 majority (mode) filter; ties keep the original label; no-data untouched."""
    valid = cls != NODATA
    counts = np.stack([ndimage.uniform_filter(((cls == k) & valid).astype(np.float32), size,
                                              mode="nearest") for k in (VEG, BUILT, OTHER)])
    best = counts.argmax(axis=0).astype(np.uint8)
    own = np.take_along_axis(counts, np.where(valid, cls, 0)[None].astype(np.int64), 0)[0]
    keep = np.isclose(own, counts.max(axis=0))
    out = np.where(keep, cls, best).astype(np.uint8)
    out[~valid] = NODATA
    return out
