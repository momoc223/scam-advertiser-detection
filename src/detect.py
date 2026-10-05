"""
Scam advertiser detection: SQL features -> rules baseline -> network signals -> ML model.

Run:  python src/generate_data.py && python src/detect.py
"""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA, SQL, FIG = ROOT / "data", ROOT / "sql", ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    for name in ["advertisers", "ads", "payments", "daily_spend"]:
        pd.read_csv(DATA / f"{name}.csv").to_sql(name, con, index=False)
    return con


def rules_score(df: pd.DataFrame) -> pd.Series:
    """Transparent baseline a reviewer could explain to an advertiser."""
    flags = pd.DataFrame({
        "new_domain": df.min_domain_age < 30,
        "cloaking": df.redirect_share >= 0.5,
        "chargebacks": df.chargeback_rate >= 0.3,
        "spend_burst": df.first_week_share >= 0.6,
        "user_reports": df.reports_per_ad >= 2,
    })
    return flags.sum(axis=1)


def network_features(con, df: pd.DataFrame) -> pd.DataFrame:
    """Link accounts that share a payment instrument or sign-up IP block."""
    links = pd.read_sql("""
        SELECT a.advertiser_id, p.payment_fingerprint, a.signup_ip_block
        FROM advertisers a JOIN payments p USING (advertiser_id)""", con)
    g = nx.Graph()
    g.add_nodes_from(links.advertiser_id)
    for col in ["payment_fingerprint", "signup_ip_block"]:
        for _, grp in links.groupby(col):
            ids = grp.advertiser_id.tolist()
            g.add_edges_from(zip(ids, ids[1:]))
    comp_size = {n: len(c) for c in nx.connected_components(g) for n in c}
    out = df.copy()
    out["cluster_size"] = out.advertiser_id.map(comp_size)
    return out


def main():
    con = load_db()
    df = pd.read_sql((SQL / "features.sql").read_text(), con)
    df = network_features(con, df).fillna(0)
    df = df.merge(pd.read_sql("SELECT advertiser_id, segment FROM advertisers", con), on="advertiser_id")
    df["rule_score"] = rules_score(df)

    features = ["first_week_share", "active_days", "n_ads", "min_domain_age",
                "avg_domain_age", "redirect_share", "n_domains", "reports_per_ad",
                "chargeback_rate", "payment_attempts", "accounts_on_card",
                "accounts_on_ip", "cluster_size", "total_spend"]

    # Split by ring so accounts from one scam ring never sit in both train and test
    groups = df.ring_id.where(df.ring_id.astype(str) != "0", "solo_" + df.advertiser_id.astype(str))
    tr, te = next(GroupShuffleSplit(test_size=.3, random_state=0).split(df, groups=groups))
    train, test = df.iloc[tr], df.iloc[te]

    model = GradientBoostingClassifier(random_state=0).fit(train[features], train.is_scam)
    test = test.assign(p_scam=model.predict_proba(test[features])[:, 1])

    # Rules baseline (block at 3+ flags) vs model (block at p >= 0.5)
    y = test.is_scam
    rules_pred = (test.rule_score >= 3).astype(int)
    model_pred = (test.p_scam >= .5).astype(int)
    legit = y == 0

    def summary(pred):
        return dict(
            precision=round(precision_score(y, pred), 3),
            recall=round(recall_score(y, pred), 3),
            legit_blocked_per_1000=round(1000 * pred[legit].mean(), 1),
        )

    results = {
        "advertisers": int(len(df)), "scam_share": round(df.is_scam.mean(), 3),
        "test_accounts": int(len(test)),
        "rules_baseline": summary(rules_pred),
        "gradient_boosting": summary(model_pred),
        "model_pr_auc": round(average_precision_score(y, test.p_scam), 3),
        "top_features": (pd.Series(model.feature_importances_, index=features)
                         .sort_values(ascending=False).head(5).round(3).to_dict()),
    }

    # Recall by segment: which kinds of scams slip through, which legit accounts get hit
    seg = test.assign(rules=rules_pred, model=model_pred).groupby("segment").agg(
        accounts=("advertiser_id", "size"), rules_flag_rate=("rules", "mean"),
        model_flag_rate=("model", "mean")).round(3)
    results["flag_rate_by_segment"] = seg.reset_index().to_dict(orient="records")

    # Investigation view: shared payment instruments
    rings = pd.read_sql((SQL / "top_rings.sql").read_text(), con)
    rings.to_csv(ROOT / "data" / "top_shared_cards.csv", index=False)

    # Figure 1: spend profile after sign-up
    con.execute("""CREATE TABLE ds AS
        SELECT s.advertiser_id, a.is_scam,
               CAST(julianday(s.spend_date) - julianday(a.created_at) AS INT) AS day,
               s.spend_usd
        FROM daily_spend s JOIN advertisers a USING (advertiser_id)""")
    prof = pd.read_sql("""SELECT is_scam, day, AVG(spend_usd) AS avg_spend
                          FROM ds WHERE day < 30 GROUP BY is_scam, day""", con)
    fig, ax = plt.subplots(figsize=(7, 4))
    for flag, label in [(0, "Legitimate"), (1, "Scam")]:
        d = prof[prof.is_scam == flag]
        ax.plot(d.day, d.avg_spend, label=label, linewidth=2)
    ax.set(title="Average daily spend after account creation",
           xlabel="Days since sign-up", ylabel="USD")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "spend_profile.png", dpi=150)

    # Figure 2: precision-recall, model vs rules
    p, r, _ = precision_recall_curve(y, test.p_scam)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(r, p, linewidth=2, label="Gradient boosting")
    rb = results["rules_baseline"]
    ax.scatter([rb["recall"]], [rb["precision"]], color="black", zorder=3, label="Rules (3+ flags)")
    ax.set(title="Precision vs recall on held-out rings", xlabel="Recall", ylabel="Precision",
           xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(frameon=False, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "precision_recall.png", dpi=150)

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("\nTop shared payment instruments:\n", rings.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
