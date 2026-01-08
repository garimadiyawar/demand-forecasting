import pandas as pd

def load_m5():
    sales = pd.read_csv("data/raw/m5/sales_train_validation.csv")
    calendar = pd.read_csv("data/raw/m5/calendar.csv")
    prices = pd.read_csv("data/raw/m5/sell_prices.csv")

    return sales, calendar, prices
