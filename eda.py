import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/sme_loan_dataset.csv")
print("✅ Dataset loaded:", df.shape)
print(df.head())

# --- Basic Info ---
print("\n--- Info ---")
print(df.info())

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Summary statistics ---")
print(df.describe())

# --- Approval Rate ---
approval_rate = df["loan_approved"].mean() * 100
print(f"\n✅ Overall Loan Approval Rate: {approval_rate:.2f}%")

# --- Sector-wise approval ---
sector_approval = df.groupby("sector")["loan_approved"].mean() * 100
print("\nApproval Rate by Sector:\n", sector_approval)

# --- Visualizations ---
sns.set(style="whitegrid")

# 1. Approval rate by sector
plt.figure(figsize=(8, 5))
sns.barplot(x=sector_approval.index, y=sector_approval.values)
plt.xticks(rotation=30)
plt.title("Approval Rate by Sector")
plt.ylabel("Approval Rate (%)")
plt.tight_layout()
plt.savefig("data/approval_rate_by_sector.png")
plt.show()

# 2. Monthly revenue vs approval
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_approved", y="monthly_revenue_ksh", data=df)
plt.yscale("log")  # revenue can be huge, log scale helps
plt.title("Monthly Revenue vs Loan Approval")
plt.tight_layout()
plt.savefig("data/revenue_vs_approval.png")
plt.show()

# 3. Collateral vs approval
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_approved", y="collateral_value_ksh", data=df)
plt.yscale("log")
plt.title("Collateral Value vs Loan Approval")
plt.tight_layout()
plt.savefig("data/collateral_vs_approval.png")
plt.show()
