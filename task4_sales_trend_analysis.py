"""Project 3: Customer Churn / Sales Trend Analysis.
Uses synthetic example data for demonstration; replace with an authorized real dataset for real business conclusions.
"""
import pandas as pd
import matplotlib.pyplot as plt

data = [
    ("2026-09-01", "Mumbai", "USB-C Cable", 120, 80, 2),
    ("2026-09-02", "Pune", "Phone Case", 199, 90, 5),
    ("2026-09-03", "Mumbai", "Wired Earphones", 299, 180, 3),
    ("2026-09-04", "Thane", "20W Charger", 499, 310, 2),
    ("2026-09-05", "Nashik", "Bluetooth Speaker", 899, 620, 1),
    ("2026-09-06", "Mumbai", "Phone Case", 199, 90, 4),
    ("2026-09-07", "Pune", "USB-C Cable", 120, 80, 3),
    ("2026-09-08", "Thane", "20W Charger", 499, 310, 2),
    ("2026-09-09", "Mumbai", "Bluetooth Speaker", 899, 620, 2),
    ("2026-09-10", "Nashik", "Wired Earphones", 299, 180, 2),
]
df = pd.DataFrame(data, columns=["date", "city", "product", "unit_price", "unit_cost", "quantity"])

# Feature engineering: parse date and derive calendar/time features.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).drop_duplicates()
df["month"] = df["date"].dt.to_period("M").astype(str)
df["weekday"] = df["date"].dt.day_name()
df["revenue"] = df["unit_price"] * df["quantity"]
df["cost"] = df["unit_cost"] * df["quantity"]
df["profit"] = df["revenue"] - df["cost"]
df["profit_margin_pct"] = (df["profit"] / df["revenue"] * 100).round(2)

print("Dataset preview:")
print(df.head().to_string(index=False))
print("\nSummary metrics")
print(f"Total revenue: INR {df['revenue'].sum():,.2f}")
print(f"Total cost: INR {df['cost'].sum():,.2f}")
print(f"Total profit: INR {df['profit'].sum():,.2f}")
print(f"Units sold: {int(df['quantity'].sum())}")

# Pivot tables required for business trend analysis.
product_pivot = pd.pivot_table(
    df, index="product", values=["quantity", "revenue", "profit"],
    aggfunc="sum"
).sort_values("revenue", ascending=False)
city_pivot = pd.pivot_table(
    df, index="city", values=["quantity", "revenue", "profit"],
    aggfunc="sum"
).sort_values("revenue", ascending=False)
daily_pivot = pd.pivot_table(
    df, index="date", values=["quantity", "revenue", "profit"], aggfunc="sum"
).sort_index()

print("\nProduct performance pivot:")
print(product_pivot.to_string())
print("\nCity performance pivot:")
print(city_pivot.to_string())
print("\nDaily trend pivot:")
print(daily_pivot.to_string())

# Visualize daily revenue trend.
daily_pivot["revenue"].plot(kind="line", marker="o", title="Daily Sample Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue (INR)")
plt.tight_layout()
plt.savefig("daily_sales_trend.png", dpi=150, bbox_inches="tight")
plt.close()

# Create a concise, data-derived recommendation summary.
top_product = product_pivot.index[0]
top_city = city_pivot.index[0]
best_profit_product = product_pivot["profit"].idxmax()
recommendations = [
    f"Highest-revenue product in this sample: {top_product}. Check stock availability before promotions.",
    f"Highest-revenue city in this sample: {top_city}. Review demand and delivery performance in this city.",
    f"Highest total-profit product in this sample: {best_profit_product}. Compare its margin and sales volume before scaling.",
    "Use a larger, real, authorized dataset before making operational or customer-level decisions."
]
print("\nActionable business recommendations:")
for item in recommendations:
    print(f"- {item}")

print("\nData note: this is synthetic demonstration data, not real company or customer data.")
