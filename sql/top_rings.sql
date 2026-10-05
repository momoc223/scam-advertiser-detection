-- Investigation query: payment instruments shared across several accounts,
-- ranked by total spend and user reports. A starting point for a ring review.
SELECT p.payment_fingerprint,
       COUNT(DISTINCT p.advertiser_id)          AS accounts,
       COUNT(DISTINCT a.signup_ip_block)        AS ip_blocks,
       ROUND(SUM(s.total_spend), 0)             AS spend_usd,
       SUM(r.reports)                           AS user_reports
FROM payments p
JOIN advertisers a USING (advertiser_id)
JOIN (SELECT advertiser_id, SUM(spend_usd) AS total_spend
      FROM daily_spend GROUP BY advertiser_id) s USING (advertiser_id)
JOIN (SELECT advertiser_id, SUM(user_reports) AS reports
      FROM ads GROUP BY advertiser_id) r USING (advertiser_id)
GROUP BY p.payment_fingerprint
HAVING COUNT(DISTINCT p.advertiser_id) >= 3
ORDER BY user_reports DESC, spend_usd DESC
LIMIT 10;
