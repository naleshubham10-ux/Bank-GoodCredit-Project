"""
fetch_from_mysql.py
-------------------
Downloads the real tables from the Bank GoodCredit MySQL database
(project_banking) into data/raw/, replacing the sample CSVs.

Credentials are read from a .env file in the project root (copy
.env.example to .env and fill in the values from the business case PDF).

    pip install pymysql sqlalchemy python-dotenv
    python src/fetch_from_mysql.py

After fetching, rerun:
    python src/feature_engineering.py
    python src/train.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

TABLES = {
    "Cust_Account": "customer_account.csv",
    "Cust_Enquiry": "customer_enquiry.csv",
    "Cust_Demographics": "customer_demographics.csv",
}


def main():
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus

    load_dotenv(os.path.join(ROOT, ".env"))
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "3306")
    user = os.environ["DB_USER"]
    pwd = quote_plus(os.environ["DB_PASSWORD"])      # password has special chars
    db = os.environ.get("DB_NAME", "project_banking")

    engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}")
    os.makedirs(RAW, exist_ok=True)
    with engine.connect() as con:
        for table, fname in TABLES.items():
            df = pd.read_sql(f"SELECT * FROM {table}", con)
            df.to_csv(os.path.join(RAW, fname), index=False)
            print(f"{table:18s} -> data/raw/{fname}  {df.shape}")


if __name__ == "__main__":
    main()
