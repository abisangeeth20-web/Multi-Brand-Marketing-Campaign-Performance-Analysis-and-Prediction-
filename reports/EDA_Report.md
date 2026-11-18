# EDA Report: Multi-Brand Marketing Campaign Performance

**Data:** Nykaa, Purplle and Tira campaign files, July 2024 – 24 June 2025.
**Rows after cleaning:** 165,230 (from 166,665 raw rows; 99.1% kept).
Figures are in `reports/figures/` and the underlying tables in `reports/eda_tables/`.

---

## 1. What the data actually is (key structural findings)

Three facts discovered during profiling shape every conclusion below.

| Finding | Evidence | Consequence |
|---|---|---|
| `Acquisition_Cost` is **cost per conversion**, not total spend | `ROI = (Revenue − Acquisition_Cost × Conversions) / (Acquisition_Cost × Conversions)` holds on 99.99% of complete rows | Total spend = `Acquisition_Cost × Conversions` (engineered as `Total_Cost`) |
| `Engagement_Score` is a funnel formula | `(Clicks + Leads + Conversions) / Impressions × 100` holds on 100% of complete rows | Missing funnel values can be recovered exactly |
| Revenue per conversion varies at random between ₹200 and ₹800 | It is uncorrelated (|r| < 0.01) with every attribute: brand, type, channel, audience, language, funnel, cost | This part of revenue cannot be forecast from campaign attributes |

Total spend per campaign also sits within a fixed band of **₹50K – ₹3L**. A high cost per conversion therefore means *fewer conversions*, not a bigger budget. That is why `Acquisition_Cost` correlates negatively with Conversions (−0.46) and Revenue (−0.41).

## 2. Data quality and cleaning summary

| Issue | Count | Treatment |
|---|---|---|
| Nulls per column | ~5% in each of the 16 columns (8,000 – 8,358 each) | see below |
| Missing `Campaign_ID` | 5,417 | Reconstructed exactly: IDs are sequential (`NY/PU/TI-CMP-<1000 + row>`) |
| Duplicate rows / IDs | 0 after reconstruction | The apparent "duplicate IDs" were all nulls |
| ROI inconsistent with Revenue / Cost | 25 | **Corrected** using the ROI identity |
| Recovered via ROI identity | 23,279 cells | Exact algebraic recovery |
| Recovered via Engagement identity | 34,345 cells | Exact algebraic recovery |
| Rows with ≥ 2 of Revenue / Cost / Conversions / ROI missing | 1,435 (0.86%) | Dropped, because the target can't be made consistent |
| Remaining funnel gaps | 2,320 | Brand-median conversion ratios; funnel monotonicity enforced |
| Duration | 8,254 | Brand median |
| Campaign type / audience / language / segment | ~8,100 – 8,300 each | Brand mode |
| Channel_Used | 8,221 | `Unknown` (no channel flag set) |
| Date | 8,018 | Sampled from the brand's observed dates (avoids an artificial spike) |

**Result:** 0 nulls, 0 duplicates, correct dtypes (dates parsed, counts as integers).

## 3. Brand performance  *(fig `eda_01`, `eda_02`, `eda_08`)*

| Brand | Campaigns | Revenue | Spend | Net profit | Portfolio ROI | Median ROI | % campaigns in profit |
|---|---|---|---|---|---|---|---|
| Nykaa | 55,053 | ₹2,840.8 Cr | ₹961.0 Cr | ₹1,879.8 Cr | **1.96** | 1.24 | 76.0% |
| Purplle | 55,079 | ₹2,829.0 Cr | ₹963.5 Cr | ₹1,865.6 Cr | 1.94 | 1.23 | 75.8% |
| Tira | 55,098 | ₹2,821.7 Cr | ₹963.3 Cr | ₹1,858.4 Cr | 1.93 | 1.24 | **76.3%** |
| **Total** | 165,230 | **₹8,491.5 Cr** | **₹2,887.7 Cr** | **₹5,603.8 Cr** | 1.94 | 1.24 | 76.0% |

* The three brands perform almost identically: revenue differs by under 0.7% and profit rate by 0.5 pp.
* Nykaa has the most revenue and profit. Tira has the highest share of profitable campaigns.
* ROI is heavily right-skewed (mean ≈ 2.7 vs median ≈ 1.24). A few campaigns return 60–79× their spend, while **~24% of campaigns lose money**.
* Monthly revenue is flat at ≈ ₹230–260 Cr per brand, with no real seasonality. June 2025 only *looks* like a drop because the data ends on 24 June.

## 4. Campaign-type performance  *(fig `eda_03`)*

| Campaign type | Median ROI | % in profit | Avg revenue |
|---|---|---|---|
| Paid Ads | **1.26** | 76.2% | ₹5.19 L |
| Social Media | 1.24 | **76.6%** | ₹5.16 L |
| SEO | 1.23 | 75.7% | ₹5.11 L |
| Email | 1.22 | 75.8% | ₹5.13 L |
| Influencer | 1.21 | 75.8% | ₹5.11 L |

* Paid Ads and Social Media lead. The best brand × type pairs are **Tira Paid Ads** (median ROI 1.30, 77.1% in profit) and **Tira / Nykaa Social Media** (77.1% / 76.7%).
* The weakest pair is **Purplle Influencer** (median ROI 1.19, 75.3% in profit).
* The whole spread is only about 2 pp, so campaign type is a *secondary* lever.

## 5. Channel effectiveness  *(fig `eda_04`, `eda_05`)*

`Channel_Used` is multi-label, so each campaign is counted once per channel it used.

| Channel | Median ROI | % in profit | Avg revenue |
|---|---|---|---|
| Instagram | **1.26** | **76.3%** | ₹5.17 L |
| Email | 1.25 | 76.2% | ₹5.17 L |
| WhatsApp | 1.23 | 76.2% | ₹5.14 L |
| Facebook | 1.23 | 76.1% | ₹5.14 L |
| YouTube | 1.22 | 76.0% | ₹5.12 L |
| Google | 1.22 | 75.8% | ₹5.12 L |

* Instagram and Email are marginally the best channels, and Google and YouTube marginally the weakest.
* **More channels don't buy more results.** Campaigns on 1, 2 or 3 channels have the same spend (≈ ₹1.75 L), revenue (≈ ₹5.14 L) and profit rate (76.0–76.1%). Adding channels spreads the same budget more thinly rather than growing it.

## 6. Audience, segment and language  *(fig `eda_09`)*

Every group falls within **75.8% – 76.2%** profit rate and median ROI 1.22 – 1.25. Premium Shoppers (target audience) and Hindi (language) are marginally ahead. These are not material drivers.

## 7. Relationships between spend, clicks, revenue and ROI  *(fig `eda_06`, `eda_07`, `eda_10`)*

* **Revenue is driven by funnel depth.** It correlates with Conversions at 0.88, Leads 0.81, Clicks 0.72 and Impressions 0.50.
* **Total spend has zero correlation with revenue** (r = 0.00). Spending more doesn't by itself bring in more revenue.
* **Cost per conversion is the strongest profitability driver** (r = −0.62 with Profit_Flag):

| Cost per conversion | < ₹100 | ₹100–200 | ₹200–300 | ₹300–400 | ₹400–500 | ₹500–600 | ₹600–800 | > ₹800 |
|---|---|---|---|---|---|---|---|---|
| % in profit | 100% | 100% | 92% | 76% | 59% | 41% | 18% | **0%** |

A campaign is profitable exactly when **revenue per conversion (₹200–800) > cost per conversion**. Below ₹200 a campaign cannot lose money, and above ₹800 it cannot make money. 17,621 campaigns (10.7%) were run at more than ₹800 per conversion and were all guaranteed to lose money.

## 8. Top and low performers

* **Top 10 by ROI** (ROI 61–79): every one spent near the floor (₹50K–61K) and earned ₹33–46 L. They pair *low cost per conversion* with *high conversion volume*, and come from all brands and types.
* **Bottom 10 by ROI** (ROI −0.97 to −0.99): each spent ₹1.8–3 L and earned under ₹10K. They had very few conversions at a very high cost per conversion.
* Example best campaign: `TI-CMP-46406` (Tira, Social Media, Google + Instagram): spent ₹50.3K and earned ₹40.4 L (ROI 79.3).

## 9. Recommendations

1. **Manage cost per conversion, not channel mix.** Set a hard ceiling of about **₹300 per conversion**: 92–100% of campaigns below it were profitable. Pause or rework any campaign trending above ₹500, and never run one above ₹800.
2. **Optimise the funnel for conversion volume.** Conversions explain most of revenue (r = 0.88). Invest in landing pages, retargeting and lead nurturing over raw reach, since impressions have the weakest link to revenue.
3. **Don't add channels by default.** Multi-channel campaigns cost the same and earn the same. Focus budget on Instagram and Email, which are marginally the strongest.
4. **Favour Paid Ads and Social Media**, especially for Tira, and review Purplle's Influencer programme.
5. **Raise average order value.** Revenue per conversion ranges from ₹200 to ₹800 and is the other half of the profit equation. Bundles, minimum-order offers and premium cross-sell move the break-even point up.
6. **Use the prediction app as a go / no-go check.** Enter planned funnel and cost figures, and use the profit probability to reject plans before launch.
