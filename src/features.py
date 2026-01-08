import pandas as pd

def melt_sales(sales):
    df = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        var_name="d",
        value_name="sales"
    )
    return df

def build_features(sales, calendar, prices):
    df = melt_sales(sales)
    df = df.merge(calendar, on="d", how="left")
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"])

    # Time features
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # Lag features
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    # Rolling stats
    df["rmean_7"] = df.groupby("id")["sales"].shift(1).rolling(7).mean()
    df["rmean_28"] = df.groupby("id")["sales"].shift(1).rolling(28).mean()

    df = df.dropna()
    return df
