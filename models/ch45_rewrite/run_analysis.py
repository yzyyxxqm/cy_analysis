from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
import numpy as np
import pandas as pd
from semopy import Model
from semopy.stats import calc_stats
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from snownlp import SnowNLP
from xgboost import XGBClassifier, XGBRegressor

try:
    import shap  # type: ignore
except Exception:
    shap = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ch45_rewrite"


def read_json(path: Path) -> list[dict[str, Any]]:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=enc) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    return data["data"]
                return [data]
            return []
        except Exception:
            continue
    return []


def to_datetime(record: dict[str, Any]) -> pd.Timestamp | pd.NaT:
    candidates = [
        "create_date_time",
        "publish_time",
        "updated_time",
        "created_time",
        "create_time",
        "time",
        "last_update_time",
        "last_modify_ts",
    ]
    for key in candidates:
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            if isinstance(value, (int, float)):
                v = int(value)
                unit = "ms" if v > 1_000_000_000_000 else "s"
                return pd.to_datetime(v, unit=unit, utc=True)
            ts = pd.to_datetime(value, utc=True)
            if pd.notna(ts):
                return ts
        except Exception:
            continue
    return pd.NaT


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@[\w\-_.]+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(record: dict[str, Any], kind: str) -> str:
    text_fields = ["content", "content_text", "title", "desc"]
    if kind == "comments":
        text_fields = ["content"]
    values = []
    for f in text_fields:
        v = record.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            values.append(s)
    return clean_text(" ".join(values))


def to_number(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def sentiment_score(text: str) -> float:
    if not text:
        return 0.5
    try:
        return float(SnowNLP(text).sentiments)
    except Exception:
        return 0.5


def load_all_records() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(DATA_ROOT.rglob("*.json")):
        platform = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        if "search_comments" in path.name:
            kind = "comments"
        elif "search_contents" in path.name:
            kind = "contents"
        else:
            continue
        records = read_json(path)
        for rec in records:
            text = extract_text(rec, kind=kind)
            if not text:
                continue
            like = to_number(
                rec.get(
                    "like_count", rec.get("liked_count", rec.get("voteup_count", 0))
                )
            )
            comment_n = to_number(
                rec.get(
                    "comment_count",
                    rec.get("comments_count", rec.get("sub_comment_count", 0)),
                )
            )
            share = to_number(rec.get("shared_count", rec.get("video_share_count", 0)))
            play = to_number(rec.get("video_play_count", 0))
            favorite = to_number(
                rec.get("video_favorite_count", rec.get("collected_count", 0))
            )
            engagement_raw = (
                like + 1.5 * comment_n + 2.0 * share + 0.05 * favorite + 0.001 * play
            )
            rows.append(
                {
                    "platform": platform,
                    "kind": kind,
                    "text": text,
                    "text_len": len(text),
                    "created_at": to_datetime(rec),
                    "like_count": like,
                    "comment_count": comment_n,
                    "share_count": share,
                    "engagement_raw": engagement_raw,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["platform", "kind", "text"]).reset_index(drop=True)
    return df


def segment_tokens(text: str) -> list[str]:
    stopwords = {
        "龙泉山",
        "成都",
        "一个",
        "这个",
        "我们",
        "你们",
        "他们",
        "就是",
        "还是",
        "真的",
        "感觉",
        "可以",
        "没有",
        "然后",
        "因为",
        "什么",
        "现在",
    }
    return [
        w
        for w in jieba.lcut(text)
        if len(w) > 1 and w not in stopwords and not w.isdigit()
    ]


def build_word_frequency(comment_df: pd.DataFrame) -> pd.DataFrame:
    bag: Counter[str] = Counter()
    for t in comment_df["text"].tolist():
        bag.update(segment_tokens(t))
    top = bag.most_common(300)
    return pd.DataFrame(top, columns=["term", "frequency"])


def build_topics(comment_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(comment_df) < 100:
        return (
            pd.DataFrame(columns=["topic", "term", "weight", "topic_size"]),
            pd.DataFrame(columns=["topic", "sentiment", "text"]),
        )

    vectorizer = TfidfVectorizer(
        tokenizer=segment_tokens, lowercase=False, max_features=3000, min_df=5
    )
    X = vectorizer.fit_transform(comment_df["text"].tolist())
    topic_count = 6 if X.shape[0] >= 600 else 4
    nmf = NMF(n_components=topic_count, random_state=42, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_

    feature_names = np.array(vectorizer.get_feature_names_out())
    rows: list[dict[str, Any]] = []
    for i, comp in enumerate(H):
        top_idx = np.argsort(comp)[-12:][::-1]
        for idx in top_idx:
            rows.append(
                {
                    "topic": f"T{i + 1}",
                    "term": feature_names[idx],
                    "weight": float(comp[idx]),
                }
            )

    comment_df = comment_df.copy()
    comment_df["topic"] = np.argmax(W, axis=1) + 1
    topic_size = comment_df["topic"].value_counts().sort_index().to_dict()
    topic_terms = pd.DataFrame(rows)
    topic_terms["topic_size"] = topic_terms["topic"].map(
        lambda x: topic_size.get(int(x[1:]), 0)
    )
    topic_assign = comment_df[["topic", "sentiment", "text"]].copy()
    topic_assign["topic"] = topic_assign["topic"].map(lambda x: f"T{x}")
    return topic_terms, topic_assign


def infer_visited_group(text: str) -> str:
    visited_re = r"去过|来过|打卡|去爬|去玩|去耍|上山"
    not_visited_re = r"没去过|没去|想去|准备去|打算去|还没去"
    if re.search(not_visited_re, text):
        return "heard_not_visited"
    if re.search(visited_re, text):
        return "heard_and_visited"
    return "unknown"


def build_ipa_proxy(topic_assign: pd.DataFrame) -> pd.DataFrame:
    work = topic_assign.copy()
    work["group"] = work["text"].map(infer_visited_group)
    work = work[work["group"] != "unknown"].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "topic",
                "importance",
                "performance",
                "quadrant",
                "sample_count",
            ]
        )

    out_rows: list[dict[str, Any]] = []
    for group, gdf in work.groupby("group"):
        gsize = len(gdf)
        perf_mean = gdf["sentiment"].mean()
        counts = gdf["topic"].value_counts()
        imp_mean = (counts / gsize).mean()
        for topic, c in counts.items():
            imp = c / gsize
            perf = float(gdf[gdf["topic"] == topic]["sentiment"].mean())
            if imp >= imp_mean and perf >= perf_mean:
                q = "keep_up"
            elif imp >= imp_mean and perf < perf_mean:
                q = "priority_fix"
            elif imp < imp_mean and perf >= perf_mean:
                q = "possible_overkill"
            else:
                q = "low_priority"
            out_rows.append(
                {
                    "group": group,
                    "topic": topic,
                    "importance": float(imp),
                    "performance": perf,
                    "quadrant": q,
                    "sample_count": int(c),
                }
            )
    return pd.DataFrame(out_rows).sort_values(
        ["group", "importance"], ascending=[True, False]
    )


def train_engagement_models(
    content_df: pd.DataFrame, output_root: Path
) -> dict[str, Any]:
    work = content_df.copy()
    work["sentiment"] = work["text"].map(sentiment_score)
    work["hashtag_count"] = work["text"].str.count("#")
    work["engagement_log"] = np.log1p(work["engagement_raw"])
    threshold = work["engagement_log"].quantile(0.70)
    work["target_high_engagement"] = (work["engagement_log"] >= threshold).astype(int)

    X = pd.DataFrame(
        {
            "text_len": work["text_len"],
            "sentiment": work["sentiment"],
            "hashtag_count": work["hashtag_count"],
        }
    )
    platform_dummies = pd.get_dummies(work["platform"], prefix="platform")
    X = pd.concat([X, platform_dummies], axis=1)
    y = work["target_high_engagement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=400, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=240,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
    }

    out: dict[str, Any] = {
        "n_samples": int(len(work)),
        "positive_rate": float(y.mean()),
        "threshold_log_engagement": float(threshold),
    }

    best_model_name = ""
    best_auc = -1.0
    best_model = None

    compare_rows: list[dict[str, Any]] = []
    best_X_test = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
        }
        out[name] = metrics
        compare_rows.append({"model": name, **metrics})
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = name
            best_model = model
            best_X_test = X_test

    out["best_model"] = best_model_name
    pd.DataFrame(compare_rows).to_csv(
        output_root / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    if best_model is not None:
        if hasattr(best_model, "feature_importances_"):
            imp = pd.Series(
                best_model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
        else:
            coef = np.abs(best_model.coef_[0])
            imp = pd.Series(coef, index=X.columns).sort_values(ascending=False)
        imp_df = imp.reset_index()
        imp_df.columns = ["feature", "importance"]
        imp_df.to_csv(
            output_root / "feature_importance.csv", index=False, encoding="utf-8-sig"
        )
        out["top_features"] = [
            {"feature": k, "importance": float(v)} for k, v in imp.head(10).items()
        ]

        if (
            shap is not None
            and best_model_name == "xgboost"
            and best_X_test is not None
        ):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(best_X_test)
            abs_mean = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame(
                {"feature": list(best_X_test.columns), "mean_abs_shap": abs_mean}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv(
                output_root / "shap_importance.csv", index=False, encoding="utf-8-sig"
            )
    return out


def run_sem_strategy_model(
    content_df: pd.DataFrame, output_root: Path
) -> dict[str, Any]:
    work = content_df.copy()
    if len(work) < 120:
        return {"warning": "insufficient samples for sem"}

    work["sentiment"] = work["text"].map(sentiment_score)
    work["engagement_log"] = np.log1p(work["engagement_raw"])
    work["hashtag_count"] = work["text"].str.count("#")
    plat_score = work.groupby("platform")["sentiment"].mean().to_dict()
    work["platform_sentiment_base"] = work["platform"].map(plat_score).fillna(0.5)

    def safe_zscore(s: pd.Series) -> pd.Series:
        std = float(s.std())
        if std <= 1e-9:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    sem_df = pd.DataFrame(
        {
            "CF": safe_zscore(work["text_len"]),
            "MTE": work["sentiment"],
            "MTP": safe_zscore(work["hashtag_count"]),
            "MTPC": safe_zscore(work["comment_count"]),
            "CWI": safe_zscore(work["engagement_log"]),
            "PWI": safe_zscore(np.log1p(work["like_count"] + 1.0)),
            "MGD": safe_zscore(np.log1p(work["share_count"] + 1.0)),
        }
    )
    sem_df = sem_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(sem_df) < 120:
        return {"warning": "insufficient clean samples for sem"}

    for col in sem_df.columns:
        if float(sem_df[col].std()) <= 1e-9:
            rng = np.random.default_rng(42)
            sem_df[col] = sem_df[col] + rng.normal(0.0, 1e-6, len(sem_df))

    model_desc = """
    CWI ~ CF + MTE + MTP + MTPC
    PWI ~ MGD + CWI
    MGD ~ CF + MTE + MTPC
    """
    model = Model(model_desc)
    try:
        model.fit(sem_df)
    except Exception as e:
        return {"warning": f"sem failed: {type(e).__name__}"}

    est = model.inspect()
    est.to_csv(
        output_root / "sem_path_estimates.csv", index=False, encoding="utf-8-sig"
    )

    stats = calc_stats(model)
    stats_df = stats.T.reset_index()
    stats_df.columns = ["metric", "value"]
    stats_df.to_csv(
        output_root / "sem_fit_indices.csv", index=False, encoding="utf-8-sig"
    )

    main_paths = est[(est["op"] == "~")][["lval", "rval", "Estimate", "p-value"]]
    main_paths = main_paths.sort_values("Estimate", ascending=False)
    main_paths.to_csv(
        output_root / "sem_main_paths.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "n_samples": int(len(sem_df)),
        "main_path_count": int(len(main_paths)),
        "top_paths": main_paths.head(8).to_dict(orient="records"),
    }


def run_spatiotemporal_proxy(df: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    work = df.copy()
    work = work.dropna(subset=["created_at"]).copy()
    if work.empty:
        return {"warning": "no timestamps for spatiotemporal proxy"}

    work["week"] = work["created_at"].dt.to_period("W").astype(str)
    weekly = (
        work.groupby(["platform", "week"], as_index=False)
        .agg(
            volume=("text", "count"),
            avg_sentiment=("sentiment", "mean"),
            total_engagement=("engagement_raw", "sum"),
        )
        .sort_values(["platform", "week"])
    )
    weekly["lag1_volume"] = weekly.groupby("platform")["volume"].shift(1)
    weekly["lag2_volume"] = weekly.groupby("platform")["volume"].shift(2)
    weekly["target_next_volume"] = weekly.groupby("platform")["volume"].shift(-1)
    model_df = weekly.dropna().copy()
    if len(model_df) < 40:
        return {"warning": "insufficient weekly rows for spatiotemporal proxy"}

    X = model_df[
        ["volume", "avg_sentiment", "total_engagement", "lag1_volume", "lag2_volume"]
    ]
    X = pd.concat([X, pd.get_dummies(model_df["platform"], prefix="platform")], axis=1)
    y = model_df["target_next_volume"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = XGBRegressor(
        n_estimators=280,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    mae = float(mean_absolute_error(y_test, pred))

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": pred})
    pred_df.to_csv(
        output_root / "spatiotemporal_proxy_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weekly.to_csv(
        output_root / "weekly_platform_volume.csv", index=False, encoding="utf-8-sig"
    )
    return {"rows": int(len(model_df)), "rmse": rmse, "mae": mae}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_all_records()
    if df.empty:
        raise SystemExit("No usable records found under data/.")

    df["sentiment"] = df["text"].map(sentiment_score)

    overview = (
        df.groupby(["platform", "kind"], as_index=False)
        .agg(
            sample_count=("text", "count"),
            avg_text_len=("text_len", "mean"),
            avg_sentiment=("sentiment", "mean"),
            median_sentiment=("sentiment", "median"),
        )
        .sort_values(["platform", "kind"])
    )
    overview.to_csv(
        OUTPUT_ROOT / "platform_overview.csv", index=False, encoding="utf-8-sig"
    )

    sentiment_bucket = pd.cut(
        df["sentiment"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["very_negative", "negative", "neutral", "positive", "very_positive"],
    )
    sentiment_dist = (
        df.assign(sentiment_bucket=sentiment_bucket)
        .groupby(["platform", "sentiment_bucket"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    sentiment_dist.to_csv(
        OUTPUT_ROOT / "sentiment_distribution.csv", index=False, encoding="utf-8-sig"
    )

    comments = df[df["kind"] == "comments"].copy()
    contents = df[df["kind"] == "contents"].copy()

    word_freq = build_word_frequency(comments)
    word_freq.to_csv(
        OUTPUT_ROOT / "word_frequency.csv", index=False, encoding="utf-8-sig"
    )

    topics, topic_assign = build_topics(comments)
    topics.to_csv(OUTPUT_ROOT / "topic_keywords.csv", index=False, encoding="utf-8-sig")
    ipa_proxy = build_ipa_proxy(topic_assign)
    ipa_proxy.to_csv(
        OUTPUT_ROOT / "ipa_proxy_summary.csv", index=False, encoding="utf-8-sig"
    )

    if len(contents) >= 80 and contents["engagement_raw"].sum() > 0:
        model_metrics = train_engagement_models(contents, OUTPUT_ROOT)
        sem_metrics = run_sem_strategy_model(contents, OUTPUT_ROOT)
    else:
        model_metrics = {
            "warning": "insufficient content samples for engagement modeling"
        }
        sem_metrics = {"warning": "insufficient content samples for sem"}

    st_metrics = run_spatiotemporal_proxy(df, OUTPUT_ROOT)

    with (OUTPUT_ROOT / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=2)

    with (OUTPUT_ROOT / "sem_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(sem_metrics, f, ensure_ascii=False, indent=2)

    with (OUTPUT_ROOT / "spatiotemporal_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(st_metrics, f, ensure_ascii=False, indent=2)

    top_examples = (
        df.sort_values("sentiment", ascending=False)[
            ["platform", "kind", "sentiment", "text"]
        ]
        .head(30)
        .copy()
    )
    top_examples["text"] = top_examples["text"].str.slice(0, 160)
    top_examples.to_csv(
        OUTPUT_ROOT / "top_positive_examples.csv", index=False, encoding="utf-8-sig"
    )

    low_examples = (
        df.sort_values("sentiment", ascending=True)[
            ["platform", "kind", "sentiment", "text"]
        ]
        .head(30)
        .copy()
    )
    low_examples["text"] = low_examples["text"].str.slice(0, 160)
    low_examples.to_csv(
        OUTPUT_ROOT / "top_negative_examples.csv", index=False, encoding="utf-8-sig"
    )

    score_samples = pd.concat(
        [
            low_examples.head(20),
            top_examples.head(20),
        ],
        axis=0,
    ).reset_index(drop=True)
    score_samples.to_csv(
        OUTPUT_ROOT / "sentiment_score_samples.csv", index=False, encoding="utf-8-sig"
    )

    print(f"records={len(df)}")
    print(f"comments={len(comments)}, contents={len(contents)}")
    print("outputs written to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
