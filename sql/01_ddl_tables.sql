-- DDL for Star Schema

CREATE TABLE IF NOT EXISTS dim_customers (
    CustomerID INTEGER PRIMARY KEY,
    Country TEXT
);

CREATE TABLE IF NOT EXISTS dim_products (
    StockCode TEXT PRIMARY KEY,
    Description TEXT
);

CREATE TABLE IF NOT EXISTS fact_sales (
    InvoiceNo TEXT,
    StockCode TEXT,
    CustomerID INTEGER,
    Quantity INTEGER,
    InvoiceDate DATETIME,
    UnitPrice REAL,
    FOREIGN KEY(CustomerID) REFERENCES dim_customers(CustomerID),
    FOREIGN KEY(StockCode) REFERENCES dim_products(StockCode)
);
