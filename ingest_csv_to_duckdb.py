# ingest_csv_to_duckdb.py
# CLI: Convert a CSV to a DuckDB table that the Streamlit app can use.
# Usage:
#   python ingest_csv_to_duckdb.py --csv /path/to/file.csv --db data/hr_attrition.duckdb --table employees

import argparse
import os
import duckdb

def main():
    parser = argparse.ArgumentParser(description="CSV -> DuckDB table")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--db", default="data/hr_attrition.duckdb", help="Output DuckDB path")
    parser.add_argument("--table", default="employees", help="Table name")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db), exist_ok=True) if os.path.dirname(args.db) else None
    con = duckdb.connect(args.db)
    print(f"[INFO] Ingesting {args.csv} -> {args.db}:{args.table}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {args.table} AS
        SELECT * FROM read_csv_auto('{args.csv}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
    """)
    rows = con.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()[0]
    cols = con.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{args.table}'").fetchone()[0]
    print(f"[OK] Created table '{args.table}' with {rows} rows and {cols} columns")

if __name__ == "__main__":
    main()
