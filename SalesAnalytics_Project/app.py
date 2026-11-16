from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_file
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from pathlib import Path
from datetime import datetime
from io import BytesIO
import sqlite3
import os
import re

import pandas as pd
import numpy as np


# ============================================================
# APP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "salesanalytics.db"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "salesanalytics-secret-key-change-me"
)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "product",
    "category",
    "customer",
    "region",
    "quantity",
    "revenue",
    "profit"
]


# ============================================================
# DATABASE CONNECTION
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():

    conn = db()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'Manager'
        );

        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            order_date TEXT,
            product TEXT,
            category TEXT,
            customer TEXT,
            region TEXT,
            quantity REAL,
            revenue REAL,
            profit REAL
        );

        CREATE TABLE IF NOT EXISTS upload_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            records INTEGER,
            columns INTEGER,
            missing INTEGER,
            duplicates INTEGER,
            status TEXT,
            uploaded_at TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            notifications INTEGER DEFAULT 1,
            compact_mode INTEGER DEFAULT 0
        );
    """)

    # Create default admin
    user = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    ).fetchone()

    if not user:

        conn.execute(
            """
            INSERT INTO users
            (username, email, password, role)
            VALUES (?, ?, ?, ?)
            """,
            (
                "admin",
                "admin@salesanalytics.com",
                generate_password_hash("admin123"),
                "Manager"
            )
        )

    conn.execute(
        "INSERT OR IGNORE INTO settings(id) VALUES(1)"
    )

    conn.commit()
    conn.close()


# ============================================================
# DEMO DATASET
# ============================================================

def create_demo_csv():

    path = DATA_DIR / "sales_data.csv"

    if path.exists():
        return path

    rng = np.random.default_rng(42)

    products = [
        ("Laptop Pro", "Electronics", 1200),
        ("Smartphone X", "Electronics", 800),
        ("Office Chair", "Furniture", 240),
        ("Wireless Headset", "Electronics", 90),
        ("Running Shoes", "Sports", 110),
        ("Smart Watch", "Electronics", 180),
        ("Backpack", "Accessories", 65),
        ("Desk Lamp", "Furniture", 45),
        ("Skin Care Kit", "Beauty", 70),
        ("Bluetooth Speaker", "Electronics", 130)
    ]

    regions = [
        "North",
        "South",
        "East",
        "West",
        "Central"
    ]

    customers = [
        "Retail",
        "Wholesale",
        "Corporate",
        "Online"
    ]

    rows = []

    order_number = 1001

    for date in pd.date_range(
        "2026-01-01",
        "2026-09-30",
        freq="D"
    ):

        for _ in range(
            int(rng.integers(1, 5))
        ):

            product, category, price = products[
                int(rng.integers(len(products)))
            ]

            quantity = int(
                rng.integers(1, 8)
            )

            revenue = round(
                quantity * price,
                2
            )

            profit = round(
                revenue * rng.uniform(
                    0.12,
                    0.30
                ),
                2
            )

            customer = customers[
                int(rng.integers(len(customers)))
            ]

            region = regions[
                int(rng.integers(len(regions)))
            ]

            rows.append([
                f"ORD{order_number}",
                date.strftime("%Y-%m-%d"),
                product,
                category,
                customer,
                region,
                quantity,
                revenue,
                profit
            ])

            order_number += 1

    df = pd.DataFrame(
        rows,
        columns=REQUIRED_COLUMNS
    )

    df.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# NORMALIZE DATASET COLUMNS
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(column).strip().lower()
        ).strip("_")
        for column in df.columns
    ]

    aliases = {

        "orderid": "order_id",
        "order_no": "order_id",
        "order_number": "order_id",

        "date": "order_date",
        "orderdate": "order_date",

        "item": "product",
        "product_name": "product",

        "customer_name": "customer",

        "sales": "revenue",
        "amount": "revenue",
        "gross_sales": "revenue",

        "units": "quantity",
        "qty": "quantity",

        "net_profit": "profit"
    }

    df.rename(
        columns={
            column: aliases.get(
                column,
                column
            )
            for column in df.columns
        },
        inplace=True
    )

    return df


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataframe(df):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    return {

        "records": len(df),

        "columns": len(df.columns),

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicates": int(
            df.duplicated().sum()
        ),

        "missing_columns": missing_columns
    }


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataframe(df):

    df = normalize_columns(df)

    info = inspect_dataframe(df)

    if info["missing_columns"]:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                info["missing_columns"]
            )
        )

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # Date conversion
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    # Numeric conversion
    for column in [
        "quantity",
        "revenue",
        "profit"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Text cleaning
    text_columns = [
        "order_id",
        "product",
        "category",
        "customer",
        "region"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # Remove duplicates
    df.drop_duplicates(
        inplace=True
    )

    # Remove invalid rows
    df.dropna(
        subset=REQUIRED_COLUMNS,
        inplace=True
    )

    # Quantity cannot be negative
    df = df[
        df["quantity"] >= 0
    ]

    # Format date
    df["order_date"] = (
        df["order_date"]
        .dt.strftime("%Y-%m-%d")
    )

    return df, info


# ============================================================
# STORE SALES DATA
# ============================================================

def replace_sales(df):

    conn = db()

    conn.execute(
        "DELETE FROM sales"
    )

    rows = [
        tuple(row)
        for row in df[
            REQUIRED_COLUMNS
        ].itertuples(
            index=False,
            name=None
        )
    ]

    conn.executemany(
        """
        INSERT INTO sales
        (
            order_id,
            order_date,
            product,
            category,
            customer,
            region,
            quantity,
            revenue,
            profit
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows
    )

    conn.commit()
    conn.close()


# ============================================================
# LOAD DEMO DATA
# ============================================================

def load_demo():

    path = create_demo_csv()

    df = pd.read_csv(path)

    clean_df, _ = clean_dataframe(df)

    replace_sales(clean_df)


# ============================================================
# GET SALES DATAFRAME
# ============================================================

def get_df():

    conn = db()

    df = pd.read_sql_query(
        """
        SELECT
            order_id,
            order_date,
            product,
            category,
            customer,
            region,
            quantity,
            revenue,
            profit
        FROM sales
        """,
        conn
    )

    conn.close()

    if df.empty:

        load_demo()

        conn = db()

        df = pd.read_sql_query(
            """
            SELECT
                order_id,
                order_date,
                product,
                category,
                customer,
                region,
                quantity,
                revenue,
                profit
            FROM sales
            """,
            conn
        )

        conn.close()

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    return df


# ============================================================
# FILTER DATA
# ============================================================

def filtered_df():

    df = get_df()

    start = request.args.get(
        "start",
        ""
    )

    end = request.args.get(
        "end",
        ""
    )

    region = request.args.get(
        "region",
        ""
    )

    product = request.args.get(
        "product",
        ""
    )

    category = request.args.get(
        "category",
        ""
    )

    customer = request.args.get(
        "customer",
        ""
    )

    if start:

        df = df[
            df["order_date"]
            >= pd.to_datetime(start)
        ]

    if end:

        df = df[
            df["order_date"]
            <= pd.to_datetime(end)
        ]

    filters = [
        ("region", region),
        ("product", product),
        ("category", category),
        ("customer", customer)
    ]

    for column, value in filters:

        if value:

            df = df[
                df[column]
                .astype(str)
                == value
            ]

    return df


# ============================================================
# ANALYTICS ENGINE
# ============================================================

def analytics(df):

    if df.empty:

        return {
            "revenue": 0,
            "orders": 0,
            "profit": 0,
            "quantity": 0,
            "growth": 0,
            "aov": 0,
            "top_region": "-",
            "top_product": "-",
            "monthly": {},
            "monthly_profit": {},
            "product": {},
            "region": {},
            "category": {},
            "customer": {},
            "table": []
        }

    monthly = (
        df.groupby(
            df["order_date"].dt.to_period("M")
        )
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique")
        )
        .sort_index()
    )

    growth = 0

    if (
        len(monthly) >= 2
        and monthly.iloc[-2]["revenue"] != 0
    ):

        growth = (
            (
                monthly.iloc[-1]["revenue"]
                -
                monthly.iloc[-2]["revenue"]
            )
            /
            monthly.iloc[-2]["revenue"]
        ) * 100

    product = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    region = (
        df.groupby("region")["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    category = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    customer = (
        df.groupby("customer")["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    table_df = (
        df.sort_values(
            "order_date",
            ascending=False
        )
        .head(15)
        .copy()
    )

    table_df["order_date"] = (
        table_df["order_date"]
        .dt.strftime("%Y-%m-%d")
    )

    return {

        "revenue": float(
            df["revenue"].sum()
        ),

        "orders": int(
            df["order_id"].nunique()
        ),

        "profit": float(
            df["profit"].sum()
        ),

        "quantity": float(
            df["quantity"].sum()
        ),

        "growth": float(
            growth
        ),

        "aov": float(
            df["revenue"].sum()
            /
            max(
                df["order_id"].nunique(),
                1
            )
        ),

        "top_region":
            str(region.index[0])
            if len(region)
            else "-",

        "top_product":
            str(product.index[0])
            if len(product)
            else "-",

        "monthly": {
            str(key): round(
                float(value),
                2
            )
            for key, value
            in monthly["revenue"].items()
        },

        "monthly_profit": {
            str(key): round(
                float(value),
                2
            )
            for key, value
            in monthly["profit"].items()
        },

        "product": {
            str(key): round(
                float(value),
                2
            )
            for key, value
            in product.items()
        },

        "region": {
            str(key): round(
                float(value),
                2
            )
            for key, value
            in region.items()
        },

        "category": {
            str(key): round(
                float(value),
                2
            )
            for key, value
            in category.items()
        },

        "customer": {
            str(key): round(
                float(value),
                2
            )
            for key, value
            in customer.items()
        },

        "table":
            table_df.to_dict(
                "records"
            )
    }


# ============================================================
# SALES FORECASTING
# ============================================================

def forecast(
    df,
    periods=6
):

    monthly = (
        df.set_index("order_date")
        .resample("MS")
        .agg(
            revenue=("revenue", "sum")
        )
        .reset_index()
    )

    if len(monthly) < 3:

        return {
            "labels": [],
            "actual": [],
            "forecast": [],
            "next": 0,
            "growth": 0,
            "confidence": "Low"
        }

    try:

        from sklearn.linear_model import (
            LinearRegression
        )

        y = (
            monthly["revenue"]
            .values
            .astype(float)
        )

        X = np.arange(
            len(y)
        ).reshape(-1, 1)

        model = LinearRegression()

        model.fit(
            X,
            y
        )

        future_x = np.arange(
            len(y),
            len(y) + periods
        ).reshape(-1, 1)

        predictions = model.predict(
            future_x
        )

        predictions = np.maximum(
            predictions,
            0
        )

        labels = [
            date.strftime("%b %Y")
            for date
            in monthly["order_date"]
        ]

        future_labels = [
            (
                monthly["order_date"].iloc[-1]
                +
                pd.DateOffset(
                    months=i
                )
            ).strftime("%b %Y")
            for i in range(
                1,
                periods + 1
            )
        ]

        labels += future_labels

        actual = [
            round(
                float(value),
                2
            )
            for value in y
        ]

        actual += [
            None
        ] * periods

        forecast_values = [
            None
        ] * len(y)

        forecast_values += [
            round(
                float(value),
                2
            )
            for value
            in predictions
        ]

        next_value = float(
            predictions[0]
        )

        forecast_growth = 0

        if y[-1] != 0:

            forecast_growth = (
                (
                    next_value
                    -
                    y[-1]
                )
                /
                y[-1]
            ) * 100

        # Calculate R²
        score = model.score(
            X,
            y
        )

        if len(y) >= 8 and score >= 0.65:

            confidence = "High"

        elif len(y) >= 5 and score >= 0.40:

            confidence = "Medium"

        else:

            confidence = "Low"

        return {

            "labels": labels,

            "actual": actual,

            "forecast":
                forecast_values,

            "next":
                round(
                    next_value,
                    2
                ),

            "growth":
                round(
                    forecast_growth,
                    2
                ),

            "confidence":
                confidence
        }

    except Exception:

        return {
            "labels": [],
            "actual": [],
            "forecast": [],
            "next": 0,
            "growth": 0,
            "confidence": "Low"
        }


# ============================================================
# SMART ALERT ENGINE
# ============================================================

def alerts_engine(a):

    alerts = []

    if a["growth"] < 0:

        alerts.append(
            (
                "High",
                "Sales decline detected",
                f"Sales growth is {a['growth']:.1f}%. "
                "Review recent low-performing segments."
            )
        )

    elif a["growth"] < 5:

        alerts.append(
            (
                "Medium",
                "Growth is slowing",
                f"Current growth is {a['growth']:.1f}%. "
                "Consider targeted campaigns."
            )
        )

    else:

        alerts.append(
            (
                "Low",
                "Positive sales momentum",
                f"Sales growth is {a['growth']:.1f}%. "
                "Maintain focus on high-performing segments."
            )
        )

    if a["region"]:

        low_region = min(
            a["region"],
            key=a["region"].get
        )

        alerts.append(
            (
                "Medium",
                "Region needs attention",
                f"{low_region} has the lowest "
                "revenue among tracked regions."
            )
        )

    if a["product"]:

        top_product = next(
            iter(a["product"])
        )

        alerts.append(
            (
                "Low",
                "Top product identified",
                f"{top_product} is currently "
                "the leading product by revenue."
            )
        )

    return alerts


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommendations_engine(
    a,
    fc
):

    recommendations = []

    if a["top_product"] != "-":

        recommendations.append(
            f"Prioritize inventory and "
            f"promotional visibility for "
            f"{a['top_product']}, the current "
            "top-revenue product."
        )

    if a["region"]:

        low_region = min(
            a["region"],
            key=a["region"].get
        )

        recommendations.append(
            f"Review pricing, demand and "
            f"sales coverage in {low_region} "
            "to improve regional performance."
        )

    if fc["next"]:

        recommendations.append(
            f"Plan near-term capacity around "
            f"the forecast revenue of "
            f"{fc['next']:,.2f}. "
            f"Forecast confidence is "
            f"{fc['confidence']}."
        )

    if a["growth"] > 0:

        recommendations.append(
            "Continue the strategies behind "
            "positive growth and compare "
            "monthly performance regularly."
        )

    else:

        recommendations.append(
            "Investigate declining months and "
            "consider targeted offers, customer "
            "re-engagement and product-level actions."
        )

    return recommendations


# ============================================================
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(function):

    @wraps(function)
    def wrapper(
        *args,
        **kwargs
    ):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        identity = request.form.get(
            "identity",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        conn = db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            OR email = ?
            """,
            (
                identity,
                identity
            )
        ).fetchone()

        conn.close()

        if (
            user
            and check_password_hash(
                user["password"],
                password
            )
        ):

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            return redirect(
                url_for("upload")
            )

        flash(
            "Invalid username/email or password.",
            "danger"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
@login_required
def upload():

    result = None

    if request.method == "POST":

        # Demo dataset
        if request.form.get("demo") == "1":

            load_demo()

            result = {
                "records": len(
                    get_df()
                ),
                "columns": 9,
                "missing_values": 0,
                "duplicates": 0,
                "status":
                    "Demo dataset loaded"
            }

        else:

            file = request.files.get(
                "dataset"
            )

            if (
                not file
                or not file.filename
            ):

                flash(
                    "Choose a CSV or XLSX file first.",
                    "warning"
                )

                return render_template(
                    "upload.html",
                    username=session["username"]
                )

            extension = Path(
                file.filename
            ).suffix.lower()

            if extension not in [
                ".csv",
                ".xlsx",
                ".xls"
            ]:

                flash(
                    "Only CSV, XLSX or XLS files are supported.",
                    "danger"
                )

                return render_template(
                    "upload.html",
                    username=session["username"]
                )

            safe_name = (
                Path(
                    file.filename
                ).name
                .replace(
                    " ",
                    "_"
                )
            )

            save_path = (
                UPLOAD_DIR
                /
                (
                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S_"
                    )
                    +
                    safe_name
                )
            )

            file.save(
                save_path
            )

            try:

                if extension == ".csv":

                    raw_df = pd.read_csv(
                        save_path
                    )

                else:

                    raw_df = pd.read_excel(
                        save_path
                    )

                clean_df, info = clean_dataframe(
                    raw_df
                )

                replace_sales(
                    clean_df
                )

                conn = db()

                conn.execute(
                    """
                    INSERT INTO upload_history
                    (
                        filename,
                        records,
                        columns,
                        missing,
                        duplicates,
                        status,
                        uploaded_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file.filename,
                        len(clean_df),
                        len(raw_df.columns),
                        info["missing_values"],
                        info["duplicates"],
                        "Processed",
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )
                )

                conn.commit()
                conn.close()

                result = {

                    "records":
                        len(clean_df),

                    "columns":
                        len(raw_df.columns),

                    "missing_values":
                        info["missing_values"],

                    "duplicates":
                        info["duplicates"],

                    "status":
                        "Dataset cleaned and processed successfully"
                }

            except Exception as error:

                result = {

                    "records": 0,
                    "columns": 0,
                    "missing_values": 0,
                    "duplicates": 0,

                    "status":
                        "Validation failed: "
                        +
                        str(error)
                }

    return render_template(
        "upload.html",
        username=session["username"],
        result=result
    )


# ============================================================
# COMMON DASHBOARD DATA
# ============================================================

def dashboard_context(active="dashboard"):

    df = filtered_df()

    analytics_data = analytics(
        df
    )

    forecast_data = forecast(
        df
    )

    alerts = alerts_engine(
        analytics_data
    )

    recommendations = recommendations_engine(
        analytics_data,
        forecast_data
    )

    all_data = get_df()

    options = {

        "regions":
            sorted(
                all_data["region"]
                .dropna()
                .astype(str)
                .unique()
            ),

        "products":
            sorted(
                all_data["product"]
                .dropna()
                .astype(str)
                .unique()
            ),

        "categories":
            sorted(
                all_data["category"]
                .dropna()
                .astype(str)
                .unique()
            ),

        "customers":
            sorted(
                all_data["customer"]
                .dropna()
                .astype(str)
                .unique()
            )
    }

    filters = {
        key: request.args.get(
            key,
            ""
        )
        for key in [
            "start",
            "end",
            "region",
            "product",
            "category",
            "customer"
        ]
    }

    return {

        "username":
            session["username"],

        "a":
            analytics_data,

        "fc":
            forecast_data,

        "alerts":
            alerts,

        "recs":
            recommendations,

        "df_count":
            len(df),

        "active":
            active,

        "filters":
            filters,

        "options":
            options
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    context = dashboard_context(
        "dashboard"
    )

    return render_template(
        "dashboard.html",
        **context
    )


# ============================================================
# SALES ANALYSIS
# ============================================================

@app.route("/analysis")
@login_required
def analysis():

    context = dashboard_context(
        "analysis"
    )

    return render_template(
        "dashboard.html",
        **context
    )


# ============================================================
# FORECASTING
# ============================================================

@app.route("/forecasting")
@login_required
def forecasting():

    context = dashboard_context(
        "forecasting"
    )

    return render_template(
        "dashboard.html",
        **context
    )


# ============================================================
# SMART ALERTS
# ============================================================

@app.route("/alerts")
@login_required
def alerts():

    context = dashboard_context(
        "alerts"
    )

    return render_template(
        "dashboard.html",
        **context
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.route("/recommendations")
@login_required
def recommendations():

    context = dashboard_context(
        "recommendations"
    )

    return render_template(
        "dashboard.html",
        **context
    )


# ============================================================
# DATA MANAGEMENT
# ============================================================

@app.route("/data-management")
@login_required
def data_management():

    conn = db()

    history = conn.execute(
        """
        SELECT *
        FROM upload_history
        ORDER BY id DESC
        """
    ).fetchall()

    count = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM sales
        """
    ).fetchone()["c"]

    conn.close()

    return render_template(
        "data_management.html",
        username=session["username"],
        history=history,
        count=count
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
@login_required
def reports():

    df = get_df()

    analytics_data = analytics(
        df
    )

    forecast_data = forecast(
        df
    )

    return render_template(
        "reports.html",
        username=session["username"],
        a=analytics_data,
        fc=forecast_data
    )


# ============================================================
# EXCEL REPORT
# ============================================================

@app.route("/reports/excel")
@login_required
def report_excel():

    df = get_df()

    analytics_data = analytics(
        df
    )

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        export_df = df.copy()

        export_df["order_date"] = (
            export_df["order_date"]
            .dt.strftime("%Y-%m-%d")
        )

        export_df.to_excel(
            writer,
            index=False,
            sheet_name="Sales Data"
        )

        kpis = pd.DataFrame([
            {
                "Metric": "Revenue",
                "Value":
                    analytics_data["revenue"]
            },
            {
                "Metric": "Orders",
                "Value":
                    analytics_data["orders"]
            },
            {
                "Metric": "Profit",
                "Value":
                    analytics_data["profit"]
            },
            {
                "Metric": "Growth %",
                "Value":
                    analytics_data["growth"]
            },
            {
                "Metric": "Average Order Value",
                "Value":
                    analytics_data["aov"]
            }
        ])

        kpis.to_excel(
            writer,
            index=False,
            sheet_name="KPIs"
        )

        pd.DataFrame(
            list(
                analytics_data[
                    "product"
                ].items()
            ),
            columns=[
                "Product",
                "Revenue"
            ]
        ).to_excel(
            writer,
            index=False,
            sheet_name="Products"
        )

        pd.DataFrame(
            list(
                analytics_data[
                    "region"
                ].items()
            ),
            columns=[
                "Region",
                "Revenue"
            ]
        ).to_excel(
            writer,
            index=False,
            sheet_name="Regions"
        )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Sales_Performance_Report.xlsx",
        mimetype=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ============================================================
# PDF REPORT
# ============================================================

@app.route("/reports/pdf")
@login_required
def report_pdf():

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import (
        getSampleStyleSheet
    )

    df = get_df()

    analytics_data = analytics(
        df
    )

    forecast_data = forecast(
        df
    )

    output = BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    story = [

        Paragraph(
            "Sales Performance Analytics Report",
            styles["Title"]
        ),

        Spacer(
            1,
            12
        )
    ]

    report_data = [

        ["Metric", "Value"],

        [
            "Revenue",
            f"{analytics_data['revenue']:,.2f}"
        ],

        [
            "Orders",
            str(
                analytics_data["orders"]
            )
        ],

        [
            "Profit",
            f"{analytics_data['profit']:,.2f}"
        ],

        [
            "Growth",
            f"{analytics_data['growth']:.2f}%"
        ],

        [
            "Average Order Value",
            f"{analytics_data['aov']:,.2f}"
        ],

        [
            "Next Forecast",
            f"{forecast_data['next']:,.2f}"
        ]
    ]

    table = Table(
        report_data,
        colWidths=[
            180,
            250
        ]
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#334155"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        table
    )

    story.append(
        Spacer(
            1,
            18
        )
    )

    story.append(
        Paragraph(
            "Top Products",
            styles["Heading2"]
        )
    )

    product_data = [
        [
            "Product",
            "Revenue"
        ]
    ]

    for product, revenue in list(
        analytics_data["product"].items()
    )[:10]:

        product_data.append(
            [
                product,
                f"{revenue:,.2f}"
            ]
        )

    product_table = Table(
        product_data,
        colWidths=[
            250,
            180
        ]
    )

    product_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#e2e8f0"
                )
            )
        ])
    )

    story.append(
        product_table
    )

    story.append(
        Spacer(
            1,
            18
        )
    )

    story.append(
        Paragraph(
            "Generated by SalesAnalytics Intelligent Dashboard.",
            styles["Normal"]
        )
    )

    document.build(
        story
    )

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Sales_Performance_Report.pdf",
        mimetype="application/pdf"
    )


# ============================================================
# API DATA
# ============================================================

@app.route("/api/data")
@login_required
def api_data():

    df = filtered_df()

    analytics_data = analytics(
        df
    )

    return jsonify({

        "kpis": {

            key:
                analytics_data[key]

            for key in [
                "revenue",
                "orders",
                "profit",
                "growth",
                "aov",
                "top_region",
                "top_product"
            ]
        },

        "monthly":
            analytics_data["monthly"],

        "product":
            analytics_data["product"],

        "region":
            analytics_data["region"],

        "category":
            analytics_data["category"],

        "customer":
            analytics_data["customer"],

        "forecast":
            forecast(df)
    })


# ============================================================
# SETTINGS
# ============================================================

@app.route(
    "/settings",
    methods=["GET", "POST"]
)
@login_required
def settings():

    if request.method == "POST":

        new_password = request.form.get(
            "new_password",
            ""
        ).strip()

        notifications = (
            1
            if request.form.get(
                "notifications"
            )
            else 0
        )

        compact_mode = (
            1
            if request.form.get(
                "compact_mode"
            )
            else 0
        )

        conn = db()

        if new_password:

            conn.execute(
                """
                UPDATE users
                SET password = ?
                WHERE id = ?
                """,
                (
                    generate_password_hash(
                        new_password
                    ),
                    session["user_id"]
                )
            )

        conn.execute(
            """
            UPDATE settings
            SET notifications = ?,
                compact_mode = ?
            WHERE id = 1
            """,
            (
                notifications,
                compact_mode
            )
        )

        conn.commit()
        conn.close()

        flash(
            "Settings updated successfully.",
            "success"
        )

        return redirect(
            url_for("settings")
        )

    conn = db()

    user = conn.execute(
        """
        SELECT username, email, role
        FROM users
        WHERE id = ?
        """,
        (
            session["user_id"],
        )
    ).fetchone()

    settings_data = conn.execute(
        """
        SELECT *
        FROM settings
        WHERE id = 1
        """
    ).fetchone()

    conn.close()

    return render_template(
        "settings.html",
        username=session["username"],
        user=user,
        settings=settings_data
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# START APPLICATION
# ============================================================

init_db()

# Load demo data if database is empty
if get_df().empty:

    load_demo()


if __name__ == "__main__":

    app.run(
        debug=True
    )