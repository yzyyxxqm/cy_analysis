from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "ch45_rewrite"
REPORT_PATH = OUT_DIR / "第4章-第5章重写稿.md"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def make_table(rows: list[str]) -> str:
    return "\n".join(rows)


def main() -> None:
    overview = pd.read_csv(OUT_DIR / "platform_overview.csv")
    sent = pd.read_csv(OUT_DIR / "sentiment_distribution.csv")
    topics = pd.read_csv(OUT_DIR / "topic_keywords.csv")
    word_freq = pd.read_csv(OUT_DIR / "word_frequency.csv")
    model_cmp = pd.read_csv(OUT_DIR / "model_comparison.csv")
    ipa = pd.read_csv(OUT_DIR / "ipa_proxy_summary.csv")
    sem_fit = pd.read_csv(OUT_DIR / "sem_fit_indices.csv")
    sem_paths = pd.read_csv(OUT_DIR / "sem_main_paths.csv")
    weekly = pd.read_csv(OUT_DIR / "weekly_platform_volume.csv")
    metrics = json.loads((OUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    sem_metrics = json.loads((OUT_DIR / "sem_metrics.json").read_text(encoding="utf-8"))
    st_metrics = json.loads(
        (OUT_DIR / "spatiotemporal_metrics.json").read_text(encoding="utf-8")
    )

    total = int(overview["sample_count"].sum())
    platform_count = overview.groupby("platform", as_index=False)["sample_count"].sum()

    pivot = sent.pivot_table(
        index="platform",
        columns="sentiment_bucket",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    sent_rows = []
    for platform, row in pivot.iterrows():
        total_p = int(row.sum())
        pos = int(row.get("positive", 0) + row.get("very_positive", 0))
        neg = int(row.get("negative", 0) + row.get("very_negative", 0))
        neutral = int(row.get("neutral", 0))
        sent_rows.append(
            {
                "platform": platform,
                "total": total_p,
                "positive_rate": pos / total_p if total_p else 0.0,
                "negative_rate": neg / total_p if total_p else 0.0,
                "neutral": neutral,
            }
        )
    sent_df = pd.DataFrame(sent_rows).sort_values("total", ascending=False)

    platform_table = make_table(
        [
            f"| {r.platform} | {int(r.sample_count)} |"
            for _, r in platform_count.sort_values(
                "sample_count", ascending=False
            ).iterrows()
        ]
    )
    sent_table = make_table(
        [
            f"| {r.platform} | {r.total} | {pct(r.positive_rate)} | {pct(r.negative_rate)} | {r.neutral} |"
            for _, r in sent_df.iterrows()
        ]
    )

    wf_table = make_table(
        [f"| {r.term} | {int(r.frequency)} |" for _, r in word_freq.head(20).iterrows()]
    )

    topic_table = []
    for topic in sorted(topics["topic"].unique()):
        sub = topics[topics["topic"] == topic].head(6)
        terms = "、".join(sub["term"].tolist())
        size = int(sub["topic_size"].iloc[0]) if len(sub) else 0
        topic_table.append(f"| {topic} | {size} | {terms} |")
    topic_table_text = make_table(topic_table)

    model_table = make_table(
        [
            f"| {r.model} | {r.roc_auc:.4f} | {r.accuracy:.4f} | {r.f1:.4f} | {r.precision:.4f} | {r.recall:.4f} |"
            for _, r in model_cmp.iterrows()
        ]
    )

    sem_fit_selected = sem_fit[
        sem_fit["metric"].isin(["CFI", "GFI", "AGFI", "NFI", "TLI", "RMSEA", "chi2"])
    ]
    sem_fit_table = make_table(
        [
            f"| {r.metric} | {float(r.value):.4f} |"
            for _, r in sem_fit_selected.iterrows()
        ]
    )

    sem_path_table = make_table(
        [
            f"| {r.lval} <- {r.rval} | {r.Estimate:.4f} | {r['p-value']:.4g} |"
            for _, r in sem_paths.head(8).iterrows()
        ]
    )

    ipa_v = ipa[ipa["group"] == "heard_and_visited"].copy()
    ipa_nv = ipa[ipa["group"] == "heard_not_visited"].copy()
    ipa_v_table = make_table(
        [
            f"| {r.topic} | {r.importance:.4f} | {r.performance:.4f} | {r.quadrant} | {int(r.sample_count)} |"
            for _, r in ipa_v.iterrows()
        ]
    )
    ipa_nv_table = make_table(
        [
            f"| {r.topic} | {r.importance:.4f} | {r.performance:.4f} | {r.quadrant} | {int(r.sample_count)} |"
            for _, r in ipa_nv.iterrows()
        ]
    )

    weekly_peak = (
        weekly.groupby("platform", as_index=False)["volume"]
        .max()
        .rename(columns={"volume": "peak_weekly_volume"})
    )
    weekly_peak_table = make_table(
        [
            f"| {r.platform} | {int(r.peak_weekly_volume)} |"
            for _, r in weekly_peak.sort_values(
                "peak_weekly_volume", ascending=False
            ).iterrows()
        ]
    )

    best_model = metrics.get("best_model", "-")

    text = f"""# 第四章 游客情感分析（重写增强版）

## 一、基于文本挖掘的分析

### （一）关于区域内生态微旅游的文本挖掘

参考文章在本节做了“数据来源说明 + 词频统计 + 词云展示 + 情感倾向判别”。本重写稿保持同类工作内容，并在方法上升级为“跨平台统一清洗 + 主题模型 + 多模型对比 + 解释增强”。

本研究共纳入 `N={total}` 条文本（评论+内容帖），样本来源如下（对应“数据来源选取表”）：

| 平台 | 样本量 |
|---|---:|
{platform_table}

### （二）数据爬取与清洗：官方报道与真实民意

清洗步骤严格对应参考文章的逻辑链条，并扩展了可复现性：
1. 字段统一：`content/title/desc/content_text` 合并为文本主字段；
2. 时间标准化：统一 `create_time/publish_time/create_date_time`；
3. 样本去重：按“平台+类型+文本”去重；
4. 文本去噪：URL、@用户、冗余空白清理；
5. 互动强度构建：`engagement_raw = like + 1.5*comment + 2*share + 0.05*favorite + 0.001*play`。

### （三）利用 SnowNLP 进行情感分析

在延续 SnowNLP 的基础上，本稿将情感分值扩展为五档分层（very_negative~very_positive），替代单阈值划分，便于后续 IPA 和策略建模联动。分层结果如下：

| 平台 | 样本量 | 正向占比 | 负向占比 | 中性样本 |
|---|---:|---:|---:|---:|
{sent_table}

可见：B 站和抖音的负向占比相对较高，贴吧与微博的正向占比更高，平台间感知差异明显，证明“多平台分层分析”是必要的。

### （四）词频统计与词云核心词替代展示（对应参考文中词频表与词云图）

参考文章给出高频词与词云，本稿同样给出高频词主表（Top20）：

| 词语 | 频次 |
|---|---:|
{wf_table}

解释：高频词同时覆盖“体验反馈（好/差）”“婚旅决策（婚礼/结婚）”“路径咨询（路线）”三类语义，说明评论并非单纯情绪发泄，而是兼具需求表达和行动信息。

## 二、消费者对生态微旅游的感观与态度分析——基于升级模型组合

### （一）数据预处理

为保持与参考文章一致，本节仍执行“编码-标准化-文本向量化”的主流程；不同点在于将模型升级为“更先进且可解释”的组合：

- 文本表示：`TF-IDF + NMF`（无监督主题）；
- 判别模型：`Logistic Regression`（稳健基线）与 `XGBoost`（高阶非线性）；
- 解释层：输出特征重要性，并为后续 SHAP 扩展预留接口。

### （二）情感分析模型的建立

参考文章采用 CNN-LSTM-Attention，本稿在当前数据条件下采用“可解释高性能替代”并给出明确目标函数：

- 高互动标签：`y_i = 1 if log(1 + engagement_i) >= Q0.70 else 0`
- 特征集合：文本长度、情感分值、话题标签数、平台哑变量。

模型对比结果（对应“模型求解结果表”）：

| 模型 | AUC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
{model_table}

最优模型为 `{best_model}`。这意味着在现有数据中，线性可解释模型仍具竞争力，但 XGBoost 作为升级模型保留了更强非线性表达能力。

### （三）情感分析模型的求解与敏感性讨论

主题结果（对应参考文“变量设置/结果表”）如下：

| 主题 | 样本数 | Top关键词 |
|---|---:|---|
{topic_table_text}

从主题结构看：
1. `T5` 体量最大，代表基础讨论流；
2. `T6` 更偏“咨询/推荐”语义，具备消费转化价值；
3. `T1/T3` 中存在体验摩擦词汇，是治理重点。

## 三、基于 IPA 模型的受众满意度分析

### （一）听说过且去过的受访者 IPA 分析

参考文章按两类人群做 IPA。本稿在缺少问卷逐题原始量表的约束下，采用“文本代理 IPA”（Importance=主题占比，Performance=主题均值情感分），并保持“两组人群”框架不变。

| 主题 | Importance | Performance | 象限 | 样本数 |
|---|---:|---:|---|---:|
{ipa_v_table}

### （二）听说过但没去过的受访者 IPA 分析

| 主题 | Importance | Performance | 象限 | 样本数 |
|---|---:|---:|---|---:|
{ipa_nv_table}

结论：两组人群都表现出 `T5` 的高重要性；未到访人群在 `T2/T3` 上表现偏弱，说明“预期管理与风险信息透明”需要加强。

---

# 第五章 生态微旅游的发展影响与刺激当地经济的决策分析（重写增强版）

## 一、生态微旅游的发展策略影响——基于 SEM + 机器学习联合框架

### （一）运用结构方程模型探究策略影响程度

参考文章在本节使用 SEM，本稿同样保留 SEM 主体，但采用可直接由社媒数据映射的观测变量构建结构：

- `CWI ~ CF + MTE + MTP + MTPC`
- `PWI ~ MGD + CWI`
- `MGD ~ CF + MTE + MTPC`

其中，`CF/MTE/MTP/MTPC` 分别由文本长度、情感分、标签强度、评论强度标准化得到；`CWI/PWI/MGD` 分别对应互动、偏好、引导代理指标。

### （二）结构方程模型的求解

拟合指标（对应参考文“模型初始拟合效果表”）：

| 指标 | 数值 |
|---|---:|
{sem_fit_table}

主要路径系数（对应“潜变量关系表”）：

| 路径 | 估计值 | p值 |
|---|---:|---:|
{sem_path_table}

解释：`PWI <- CWI` 路径最强且显著，说明“互动强度”对“消费偏好表达”有直接驱动；`CWI <- MTPC` 显著，说明评论互动环境会影响总体意愿强度。

SEM样本规模：`{sem_metrics.get("n_samples", "NA")}`。

## 二、生态微旅游刺激当地经济的发展决策——基于时序代理预测

### （一）人流分布时空预测模型的建立

参考文章使用图卷积双重注意力网络做时空预测。本稿在现有数据不含轨迹坐标的现实约束下，采用“可复现替代框架”：

1. 按平台-周构建时序面板（volume、avg_sentiment、total_engagement）；
2. 使用 `lag1/lag2` 形成时序依赖；
3. 用 `XGBRegressor` 预测下周流量代理值。

### （二）模型求解

预测效果：
- 有效训练行数：`{st_metrics.get("rows", "NA")}`
- RMSE：`{st_metrics.get("rmse", float("nan")):.4f}`
- MAE：`{st_metrics.get("mae", float("nan")):.4f}`

峰值周流量（平台维）：

| 平台 | 峰值周流量 |
|---|---:|
{weekly_peak_table}

策略含义：峰值平台应优先做交通疏导、现场秩序和商户容量冗余；非峰值平台可做精准种草和产品预售转化。

## 三、定性挖掘问题：生态、经济、社会三维治理建议

### （一）生态领域
- 对“高频-低表现”主题实施分时预约、重点区限流和脆弱点监测；
- 把夜间活动热点与环境承载联动，避免旺季生态透支。

### （二）经济领域
- 按主题热度推进差异化产品包（夜游、露营、亲子、轻徒步）；
- 用“高互动转化率 + 负向主题下降率”作为招商与经营者考核指标。

### （三）社会领域
- 建立平台舆情日清单，形成“发现-响应-复盘”闭环；
- 旺季跨部门联动，降低居民生活空间与游客活动空间冲突。

---

## 与参考文章的对应关系（本稿已完成）

1. 文本挖掘：已给出数据来源、词频统计、情感分层；
2. 情感模型：已完成模型建立、求解、指标对比与解释；
3. IPA分析：已按“去过/未去过”两类人群给出结果；
4. 发展影响：已保留 SEM 分析并给出拟合与路径结果；
5. 决策预测：已给出时序预测模型、误差指标和平台峰值判断。

## 可复现文件

- 代码：`models/ch45_rewrite/run_analysis.py`、`models/ch45_rewrite/generate_chapters_45.py`
- 文档：`outputs/ch45_rewrite/第4章-第5章重写稿.md`
- 关键结果：
  - `outputs/ch45_rewrite/word_frequency.csv`
  - `outputs/ch45_rewrite/topic_keywords.csv`
  - `outputs/ch45_rewrite/model_comparison.csv`
  - `outputs/ch45_rewrite/ipa_proxy_summary.csv`
  - `outputs/ch45_rewrite/sem_fit_indices.csv`
  - `outputs/ch45_rewrite/sem_main_paths.csv`
  - `outputs/ch45_rewrite/spatiotemporal_metrics.json`
"""

    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
