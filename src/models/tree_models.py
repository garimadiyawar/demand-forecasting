import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

FEATURES = [
    "dow", "week", "month",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rmean_7", "rmean_28",
    "sell_price"
]

def train_lightgbm(df):
    X = df[FEATURES]
    y = df["sales"]

    tscv = TimeSeriesSplit(n_splits=3)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8
    )

    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

    return model
