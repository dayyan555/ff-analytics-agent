CREATE DATABASE IF NOT EXISTS marketing;
CREATE TABLE IF NOT EXISTS marketing.campaigns (
  campaign_id UInt16, campaign_name String, channel LowCardinality(String),
  country LowCardinality(String), objective LowCardinality(String),
  start_date Date, end_date Nullable(Date)
) ENGINE = MergeTree ORDER BY campaign_id;
CREATE TABLE IF NOT EXISTS marketing.ad_spend (
  date Date, campaign_id UInt16, device LowCardinality(String),
  spend Decimal(12,2), impressions UInt32, clicks UInt32
) ENGINE = MergeTree ORDER BY (campaign_id, date, device);
CREATE TABLE IF NOT EXISTS marketing.purchases (
  purchase_id UInt32, purchased_at DateTime('UTC'), campaign_id UInt16,
  device LowCardinality(String), revenue Decimal(12,2)
) ENGINE = MergeTree ORDER BY (campaign_id, purchased_at);
