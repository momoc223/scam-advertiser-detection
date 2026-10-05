-- Advertiser-level risk features, built in SQL from raw tables.
-- Tables: advertisers, ads, payments, daily_spend

WITH spend AS (
    SELECT s.advertiser_id,
           SUM(s.spend_usd)                                         AS total_spend,
           SUM(CASE WHEN julianday(s.spend_date) - julianday(a.created_at) < 7
                    THEN s.spend_usd ELSE 0 END)                    AS first_week_spend,
           COUNT(*)                                                 AS active_days
    FROM daily_spend s
    JOIN advertisers a USING (advertiser_id)
    GROUP BY s.advertiser_id
),
ad_stats AS (
    SELECT advertiser_id,
           COUNT(*)                                          AS n_ads,
           MIN(domain_age_days)                              AS min_domain_age,
           AVG(domain_age_days)                              AS avg_domain_age,
           AVG(CASE WHEN landing_redirects >= 2 THEN 1.0 ELSE 0 END) AS redirect_share,
           COUNT(DISTINCT landing_domain)                    AS n_domains,
           SUM(user_reports) * 1.0 / COUNT(*)                AS reports_per_ad
    FROM ads
    GROUP BY advertiser_id
),
card_sharing AS (
    SELECT payment_fingerprint, COUNT(*) AS accounts_on_card
    FROM payments GROUP BY payment_fingerprint
),
ip_sharing AS (
    SELECT signup_ip_block, COUNT(*) AS accounts_on_ip
    FROM advertisers GROUP BY signup_ip_block
)
SELECT a.advertiser_id,
       a.is_scam,
       a.ring_id,
       sp.total_spend,
       sp.first_week_spend / NULLIF(sp.total_spend, 0)      AS first_week_share,
       sp.active_days,
       ad.n_ads,
       ad.min_domain_age,
       ad.avg_domain_age,
       ad.redirect_share,
       ad.n_domains,
       ad.reports_per_ad,
       p.chargebacks * 1.0 / NULLIF(p.payment_attempts, 0)  AS chargeback_rate,
       p.payment_attempts,
       cs.accounts_on_card,
       ips.accounts_on_ip
FROM advertisers a
JOIN spend        sp  USING (advertiser_id)
JOIN ad_stats     ad  USING (advertiser_id)
JOIN payments     p   USING (advertiser_id)
JOIN card_sharing cs  ON cs.payment_fingerprint = p.payment_fingerprint
JOIN ip_sharing   ips ON ips.signup_ip_block   = a.signup_ip_block;
