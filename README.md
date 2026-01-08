# Demand Forecasting with LightGBM (Retail Inventory Optimization)

## 1. Problem Statement
Retailers must balance product availability with inventory holding costs. Overstocking leads to capital lock-in and wastage, while understocking causes lost sales. This project builds a **SKU-level demand forecasting system** and demonstrates how improved forecast accuracy translates into **measurable inventory cost savings**.

The goal is not only to predict demand, but to show **operational impact** using realistic inventory assumptions.

---

## 2. Dataset

This project uses the **M5 Forecasting – Accuracy dataset** (Kaggle), a widely used benchmark for retail demand forecasting.

**Core files used:**
- `sales_train_validation.csv` – historical daily unit sales per SKU
- `calendar.csv` – calendar features, events, SNAP flags
- `sell_prices.csv` – historical sell prices

The dataset contains daily sales for ~30,000 SKUs across multiple stores and categories. For computational realism, a representative SKU subset is used.

---

## 3. Project Structure

```
demand-forecasting/
├── data/
│   ├── raw/m5/                # Original M5 CSV files
│   └── processed/             # Feature-engineered parquet files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baselines.ipynb
│   ├── 04_modeling_advanced.ipynb
│   └── 05_scenario_and_roi.ipynb
├── src/                        # Reusable pipeline components
│   ├── data_loader.py
│   ├── features.py
│   ├── models/
│   └── utils/
├── dashboards/                 # Streamlit dashboard (optional)
├── api/                        # FastAPI service (optional)
└── README.md
```

---

## 4. Modeling Approach

### Baselines
Two strong classical benchmarks were implemented:
- **Naive forecast** (yesterday = today)
- **Seasonal naive forecast** (same weekday last week)

These baselines establish a realistic lower bound for performance, which is crucial in retail demand forecasting.

### Advanced Model
A **global LightGBM regressor** was trained across all SKUs using:
- Lag features (1, 7, 14, 28 days)
- Rolling means (7, 28 days)
- Calendar effects (weekday, events, SNAP flags)
- Price information

Categorical identifiers (SKU, store, category) were handled using **LightGBM’s native categorical feature support**, avoiding one-hot encoding.

---

## 5. Evaluation Strategy

- **Time-based split** (no random leakage)
- Final **28-day horizon** used for validation
- Metric: **Mean Absolute Error (MAE)**

This mirrors real-world forecasting, where future demand must be predicted without access to future information.

---

## 6. Results

| Model            | MAE   |
|------------------|-------|
| Naive            | ~1.095 |
| Seasonal Naive   | ~1.097 |
| LightGBM (Global)| **~0.908** |

**Performance gain:**
- ~17% MAE improvement over seasonal naive baseline

This is a strong result given the intermittent and noisy nature of SKU-level retail demand.

---

## 7. Scenario Simulation & Business Impact

To translate accuracy into value, the trained model was frozen and used to simulate a **28-day forecast horizon** under realistic operational assumptions:

- Fixed lead time
- Current inventory levels
- Safety stock proportional to forecasted demand

### Pilot-Scale Impact
- Estimated annual inventory savings: **₹35,000+** on a limited SKU subset

### Scaling Insight
Since this analysis covers a small fraction of the full SKU universe, savings scale approximately linearly. At full portfolio scale, this implies **order-of-magnitude higher annual impact**.

This demonstrates how forecast improvements directly inform better replenishment decisions.

---

## 8. Key Takeaways

- Naive baselines are strong and difficult to beat in retail
- Global tree-based models perform well for sparse demand
- Forecast accuracy alone is insufficient — business impact matters
- Conservative assumptions lead to defensible, realistic savings estimates

---

## 9. Reproducibility

### Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

### Execution Order
1. Run notebooks `01` → `05` sequentially
2. Generated features are stored in `data/processed/`
3. Optional: launch dashboard or API

---

## 10. Future Improvements

- Probabilistic forecasting for service-level optimization
- Multi-horizon forecasting models
- Store-level inventory constraints
- Automated retraining and monitoring

---

## 11. Why This Project

This project was designed to mirror **real-world ML systems**, balancing:
- statistical rigor
- engineering discipline
- and business relevance

It demonstrates not just model-building, but **decision-making with models** — the core requirement of applied machine learning roles.

