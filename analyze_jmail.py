import duckdb

conn = duckdb.connect()
df = conn.sql("""
  SELECT * FROM read_parquet('https://data.jmail.world/v1/emails.parquet')
  LIMIT 100
""").df()
print(df)