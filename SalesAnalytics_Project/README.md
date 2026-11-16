# SalesAnalytics – Intelligent Sales Performance Analytics Dashboard

A Flask-based business analytics dashboard for uploading, cleaning, analyzing and forecasting sales data.

## Features
- Login and role-ready authentication
- CSV/XLSX upload with validation and preprocessing
- Demo dataset
- KPI dashboard and interactive Chart.js visualizations
- Product, region, category and customer analysis
- Date/product/region filters
- ML-based monthly revenue forecasting using Scikit-learn Linear Regression
- Smart performance alerts
- Business recommendations
- Excel and PDF reports
- Dataset upload history
- Settings and password change

## Run
```bash
python -m venv venv
venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

Demo login: `admin` / `admin123`

## Dataset columns
`order_id, order_date, product, category, customer, region, quantity, revenue, profit`

The current version uses SQLite for a portable demo. MySQL can be introduced later without changing the dashboard workflow.
