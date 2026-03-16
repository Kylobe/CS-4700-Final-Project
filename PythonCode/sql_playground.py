import sqlite3

db_path = "stockfish_label_cache.db"
conn = sqlite3.connect(db_path)
n = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
print("rows:", n)


