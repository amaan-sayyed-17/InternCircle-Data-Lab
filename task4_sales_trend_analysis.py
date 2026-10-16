"""Task 4: analyze sample e-commerce sales trends."""
import pandas as pd
import matplotlib.pyplot as plt

# Clearly labelled synthetic example data for demonstration.
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
df["date"] = pd.to_datetime(df["date"])
df["revenue"] = df["unit_price"] * df["quantity"]
df["cost"] = df["unit_cost"] * df["quantity"]
df["profit"] = df["revenue"] - df["cost"]

print("Dataset preview:")
print(df.head())
print("\nTotal revenue:", round(df["revenue"].sum(), 2))
print("Total profit:", round(df["profit"].sum(), 2))
print("Units sold:", int(df["quantity"].sum()))
print("\nRevenue by product:")
print(df.groupby("product")["revenue"].sum().sort_values(ascending=False))
print("\nRevenue by city:")
print(df.groupby("city")["revenue"].sum().sort_values(ascending=False))

daily = df.groupby("date", as_index=False)["revenue"].sum()
daily.plot(x="date", y="revenue", kind="line", marker="o", legend=False, title="Daily Sample Revenue")
plt.ylabel("Revenue (INR)")
plt.tight_layout()
plt.savefig("daily_sales_trend.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nNote: this is synthetic demonstration data, not real customer/company data.")
