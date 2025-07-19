# 🏦 DeFi Wallet Credit Scoring — Aave V2

This project uses raw transaction-level data from Aave V2 to generate credit scores (0–1000) for individual wallets, based on their behavioral patterns.

### ✅ What It Does

- Processes user-level DeFi transactions (`deposit`, `borrow`, `repay`, etc.)
- Engineers relevant features (activity days, tx types, liquidation risk)
- Trains a machine learning model (Random Forest) to generate credit scores
- Scores range from **0 (high risk)** to **1000 (safe/reliable)**

### 💡 Feature Engineering

| Feature         | Description                              |
|----------------|------------------------------------------|
| `total_tx`      | Total transactions by wallet             |
| `unique_actions`| Number of unique action types            |
| `days_active`   | Days between first and last transaction  |
| `tx_per_day`    | Transactions per active day              |
| `deposits`      | Number of deposit actions                |
| `borrows`       | Number of borrow actions                 |
| `repays`        | Number of repay actions                  |
| `redeems`       | Number of redeemunderlying actions       |
| `liquidations`  | Number of times wallet got liquidated    |

### 🧠 Model Architecture

- `RandomForestRegressor`: Trained on behavioral features
- Output risk scores mapped to [0–1000] using `MinMaxScaler`

### 📁 Input Format

Place your JSON file in `data/tx_data.json`. Each record must include:

```json
{
  "userWallet": "0x123...",
  "timestamp": 1688758273,
  "action": "deposit",
  ...
}
