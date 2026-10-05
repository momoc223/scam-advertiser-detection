"""
Generate a simulated ads-platform dataset with labelled scam advertisers.

The data is synthetic. It is modelled on abuse patterns that ads platforms
describe publicly, and it deliberately includes hard cases on both sides:

Scam advertisers
  * burst rings   - account farms sharing cards and IP blocks, new domains,
                    cloaked landing pages, heavy early spend
  * sleeper rings - shared cards, but aged domains and a slow, legit-looking ramp
  * solo scams    - single accounts with only some of the signals
Legitimate advertisers
  * regular businesses
  * new small businesses - new domains and launch-week spending spikes
  * agency-managed accounts - many accounts on one card and IP block

No real people or companies are represented.
Outputs (in data/): advertisers.csv, ads.csv, payments.csv, daily_spend.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "BR", "NG", "PH", "VN"]
START = pd.Timestamp("2026-01-01")


class Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.adv, self.ads, self.pays, self.spend = [], [], [], []
        self.next_id = 0

    def account(self, *, is_scam, kind, ring, card, ip, domain_root, domain_age,
                redirect_p, report_rate, chargeback_rate, burst, base_spend, days):
        r = self.rng
        self.next_id += 1
        aid = self.next_id
        created = START + pd.Timedelta(days=int(r.integers(0, 240)))
        self.adv.append(dict(advertiser_id=aid, created_at=created.date(),
                             country=r.choice(COUNTRIES), signup_ip_block=ip,
                             is_scam=int(is_scam), segment=kind, ring_id=ring))
        attempts = int(r.integers(1, 6))
        self.pays.append(dict(advertiser_id=aid, payment_fingerprint=card,
                              payment_attempts=attempts,
                              chargebacks=int(r.binomial(attempts, chargeback_rate))))
        for _ in range(int(r.integers(1, 10))):
            self.ads.append(dict(
                advertiser_id=aid,
                landing_domain=f"{domain_root}-{r.integers(1, 3)}.example",
                domain_age_days=int(max(1, r.normal(*domain_age))),
                landing_redirects=int(r.binomial(4, redirect_p)),
                user_reports=int(r.poisson(report_rate))))
        for d in range(days):
            shape = np.exp(-d / burst) if burst else min(1, (d + 1) / 20)
            amt = base_spend * shape * r.lognormal(0, .5)
            self.spend.append(dict(advertiser_id=aid,
                                   spend_date=(created + pd.Timedelta(days=d)).date(),
                                   spend_usd=round(float(amt), 2)))


def make_data(n_legit, n_rings, seed):
    b = Builder(seed)
    r = b.rng
    # --- legitimate advertisers -------------------------------------------
    for i in range(n_legit):
        kind = r.choice(["regular", "new_business"], p=[.88, .12])
        new = kind == "new_business"
        b.account(is_scam=False, kind=kind, ring=None, card=f"card_L{i}",
                  ip=f"ip_L{r.integers(0, n_legit * 2)}", domain_root=f"biz{i}",
                  domain_age=(20, 15) if new else (1200, 900),
                  redirect_p=.15, report_rate=r.choice([.05, .5, 1.5], p=[.85, .12, .03]),
                  chargeback_rate=r.choice([0, .15], p=[.9, .1]),
                  burst=r.choice([None, 8]) if new else None,
                  base_spend=r.uniform(20, 300), days=int(r.integers(15, 90)))
    for g in range(n_legit // 150):                       # marketing agencies
        card, ip = f"card_AG{g}", f"ip_AG{g}"
        for k in range(int(r.integers(4, 15))):
            b.account(is_scam=False, kind="agency_client", ring=None, card=card, ip=ip,
                      domain_root=f"client{g}_{k}", domain_age=(900, 700),
                      redirect_p=.2, report_rate=.1, chargeback_rate=.02, burst=None,
                      base_spend=r.uniform(100, 600), days=int(r.integers(20, 90)))
    # --- scam advertisers ---------------------------------------------------
    for g in range(n_rings):
        sleeper = r.random() < .3
        size = int(r.integers(3, 10))
        cards = [f"card_R{g}_{k}" for k in range(int(r.integers(1, 4)))]
        ips = [f"ip_R{g}_{k}" for k in range(int(r.integers(1, 4)))]
        for _ in range(size):
            b.account(is_scam=True, kind="sleeper_ring" if sleeper else "burst_ring",
                      ring=f"ring_{g}", card=r.choice(cards), ip=r.choice(ips),
                      domain_root=f"offer{g}",
                      domain_age=(600, 500) if sleeper else (40, 40),
                      redirect_p=.25 if sleeper else .55,
                      report_rate=.3 if sleeper else 1.2,
                      chargeback_rate=.1 if sleeper else .35,
                      burst=None if sleeper else 6,
                      base_spend=r.uniform(100, 500) if sleeper else r.uniform(300, 1500),
                      days=int(r.integers(10, 60) if sleeper else r.integers(4, 25)))
    for s in range(n_rings * 2):                          # solo scams
        b.account(is_scam=True, kind="solo_scam", ring=None, card=f"card_S{s}",
                  ip=f"ip_L{r.integers(0, n_legit * 2)}", domain_root=f"deal{s}",
                  domain_age=(150, 200), redirect_p=.3, report_rate=.6,
                  chargeback_rate=.2, burst=r.choice([None, 7]),
                  base_spend=r.uniform(100, 800), days=int(r.integers(5, 40)))
    return dict(advertisers=pd.DataFrame(b.adv), ads=pd.DataFrame(b.ads),
                payments=pd.DataFrame(b.pays), daily_spend=pd.DataFrame(b.spend))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--legit", type=int, default=6000)
    p.add_argument("--rings", type=int, default=45)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    for name, df in make_data(a.legit, a.rings, a.seed).items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")
