
## 💼 SME Loan Approval Predictor

A **machine learning web app** that predicts the probability of loan approval for **Kenyan SMEs** based on their financial health, credit history, and business profile.

✅ **Tech Stack:**

* Python (Pandas, NumPy, Scikit-learn)
* RandomForest ML Model
* Streamlit for interactive UI
* Matplotlib & Seaborn for visualizations

✅ **Use Case:**

* Helps **banks, microfinance institutions, and SACCOs** assess SME loan applications.
* Guides **SME owners** on how to improve their chances of approval.



### 🚀 Features

✔ Predict loan approval probability in **Kenyan Shillings (KES)**
✔ Business sector & financial inputs tailored for **Kenyan SMEs**
✔ Displays **feature importance** – what matters most for approval
✔ Recommendations on how to improve approval chances
✔ Interactive **Streamlit web app**




### 📂 Project Structure

```
SME_loan_scoring/
│── app/                 # Streamlit app
│   └── app.py
│── model/               # Trained ML model (RandomForest)
│   └── sme_loan_model.pkl
│── notebooks/           # Jupyter notebooks for analysis
│── train_model.py       # Training script
│── requirements.txt     # Project dependencies
└── README.md            # Project documentation




### 🔧 Installation & Usage

1️⃣ **Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/SME_loan_scoring.git
cd SME_loan_scoring


2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt


3️⃣ **Run the Streamlit app**

```bash
python -m streamlit run app/app.py




### 🧠 Model Details

* **Algorithm:** RandomForestClassifier
* **Training data:** Synthetic SME loan dataset (business age, revenue, expenses, credit history, collateral, etc.)
* **Target:** Loan approval (Approved/Rejected)



### 🌍 Real-World Impact

* **Banks & MFIs:** Quick pre-screening for SME loan eligibility
* **Entrepreneurs:** Insights into improving loan approval chances
* **Policymakers:** Data-driven SME financing insights



### 📜 License

MIT License – free to use & modify.


### 👨‍💻 Author

**Bill Nzuki**
📧 [billnzuki@gmail.com](mailto:billnzuki@gmail.com)


