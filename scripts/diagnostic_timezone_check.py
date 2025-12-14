# run in a small script / REPL
import pyarrow.parquet as pq
import pandas as pd
p = r"D:\trading_data\2024\01\20240103.parquet"   # example path
df = pd.read_parquet(p)
print("cols:", df.columns.tolist())
print("rows:", len(df))
print(df[['symbol','timestamp']].head(10))
print("timestamp dtype:", df['timestamp'].dtype)
print("min/max ts:", df['timestamp'].min(), df['timestamp'].max())
