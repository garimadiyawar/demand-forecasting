def naive_forecast(df):
    return df.groupby("id")["sales"].shift(1)

def seasonal_naive(df, season=7):
    return df.groupby("id")["sales"].shift(season)
