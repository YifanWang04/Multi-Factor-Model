import yfinance as yf
import pandas as pd

ticker = "AAPL"

df = yf.download(
    ticker,
    period="1d",
    interval="1m",
    prepost=True,
    progress=False
)

last_time = df.index[-1]
now = pd.Timestamp.now(tz=last_time.tz)

print("Last bar time:", last_time)
print("Now:", now)
print("Delay:", now - last_time)
