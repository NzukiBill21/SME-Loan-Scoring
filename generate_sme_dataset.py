import pandas as pd
import numpy as np
import random

np.random.seed(42)

# Kenyan SME sectors
sectors = ["Retail", "Agriculture", "Transport (Boda Boda)", "Tech Startup", "Hospitality", "Manufacturing", "Services"]

# Function to generate realistic SME data
def generate_sme_data(n=5000):
    data = []
    for _ in range(n):
        business_age = np.random.randint(1, 15)  # in years
        monthly_revenue = np.random.randint(20000, 2000000)  # in Ksh
        monthly_expenses = monthly_revenue * np.random.uniform(0.3, 0.8)  # 30%-80% of revenue
        employees = np.random.randint(1, 50)
        collateral_value = np.random.randint(0, 1000000)  # Ksh
        requested_loan = np.random.randint(50000, 2000000)  # Ksh
        sector = random.choice(sectors)
        credit_history_score = np.random.randint(300, 850)  # like a credit score

        # Basic approval logic
        approval_chance = (
            (business_age / 15) * 0.2 +
            (monthly_revenue / 2000000) * 0.3 +
            (collateral_value / 1000000) * 0.2 +
            (credit_history_score / 850) * 0.3
        )

        # If approval_chance > 0.5 → approved
        approved = 1 if approval_chance > 0.5 else 0

        data.append([
            business_age,
            sector,
            monthly_revenue,
            monthly_expenses,
            employees,
            collateral_value,
            requested_loan,
            credit_history_score,
            approved
        ])
    
    df = pd.DataFrame(data, columns=[
        "business_age_yrs",
        "sector",
        "monthly_revenue_ksh",
        "monthly_expenses_ksh",
        "num_employees",
        "collateral_value_ksh",
        "requested_loan_ksh",
        "credit_history_score",
        "loan_approved"
    ])
    return df

# Generate dataset
df_sme = generate_sme_data(5000)

# Save to CSV
df_sme.to_csv("data/sme_loan_dataset.csv", index=False)
print("✅ SME Loan Dataset generated → data/sme_loan_dataset.csv")
print(df_sme.head())
print("\nApproval Rate:", df_sme["loan_approved"].mean() * 100, "%")
