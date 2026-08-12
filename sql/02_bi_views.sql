-- BI Views for Data Marts

-- 1. Sales Summary by Customer and Product
CREATE VIEW IF NOT EXISTS v_sales_summary AS
SELECT 
    f.InvoiceNo,
    f.InvoiceDate,
    c.CustomerID,
    c.Country,
    p.StockCode,
    p.Description,
    f.Quantity,
    f.UnitPrice,
    (f.Quantity * f.UnitPrice) as TotalAmount
FROM fact_sales f
JOIN dim_customers c ON f.CustomerID = c.CustomerID
JOIN dim_products p ON f.StockCode = p.StockCode;

-- 2. Customer RFM Base Data
CREATE VIEW IF NOT EXISTS v_customer_rfm_base AS
SELECT 
    CustomerID,
    MAX(InvoiceDate) as LastPurchaseDate,
    COUNT(DISTINCT InvoiceNo) as Frequency,
    SUM(Quantity * UnitPrice) as MonetaryValue
FROM fact_sales
WHERE Quantity > 0 -- Exclude returns
GROUP BY CustomerID;

-- 3. Monthly Cohort Analysis Base
CREATE VIEW IF NOT EXISTS v_cohort_base AS
WITH FirstPurchase AS (
    SELECT CustomerID, MIN(DATE(InvoiceDate, 'start of month')) as CohortMonth
    FROM fact_sales
    GROUP BY CustomerID
),
MonthlyActivity AS (
    SELECT DISTINCT CustomerID, DATE(InvoiceDate, 'start of month') as ActivityMonth
    FROM fact_sales
)
SELECT 
    f.CohortMonth,
    m.ActivityMonth,
    COUNT(DISTINCT m.CustomerID) as ActiveCustomers
FROM MonthlyActivity m
JOIN FirstPurchase f ON m.CustomerID = f.CustomerID
GROUP BY f.CohortMonth, m.ActivityMonth;
