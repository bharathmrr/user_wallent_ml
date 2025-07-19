# user_wallent_ml
# Aave V2 Credit Scoring Engine

This project generates credit scores for wallets based on their historical transactions on the Aave V2 protocol. Scores range from 0 (very risky) to 1000 (very reliable).

## 📊 Method

We derive behavioral features for each wallet from raw transaction data and apply a rule-based weighted scoring model. Key features used:

- Total Transactions
- Total Deposit Amount
- Total Borrow Amount
- Total Repay Amount
- Number of Unique Transaction Types
- Borrow/Deposit Ratio (penalized if high)
- Repay/Borrow Ratio (rewarded if high)

## ⚙️ Architecture

- Input: `sample_transactions.json`
- Output: Wallet scores CSV and distribution graph
- Scoring: Normalized + Weighted Sum Model
- One-Step Execution: Run `score_generator.py`

## 📈 Scoring Logic

```python
score = 0.15 * total_txn + 0.2 * total_deposit + 0.25 * repay_ratio
        - 0.1 * borrow_deposit_ratio + 0.2 * txn_types + 0.1 * avg_amount
