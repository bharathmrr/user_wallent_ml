import json
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def engineer_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors='coerce')

    df = df.dropna(subset=["timestamp"])
    df = df[df["action"].notnull()]

    grouped = df.groupby("userWallet")

    features = []
    for wallet, group in grouped:
        total_tx = len(group)
        action_counts = group["action"].value_counts().to_dict()
        days_active = (group["timestamp"].max() - group["timestamp"].min()).days + 1
        tx_per_day = total_tx / days_active if days_active > 0 else total_tx

        feature = {
            "wallet": wallet,
            "total_tx": total_tx,
            "unique_actions": group["action"].nunique(),
            "days_active": days_active,
            "tx_per_day": tx_per_day,
            "deposits": action_counts.get("deposit", 0),
            "borrows": action_counts.get("borrow", 0),
            "repays": action_counts.get("repay", 0),
            "redeems": action_counts.get("redeemunderlying", 0),
            "liquidations": action_counts.get("liquidationcall", 0),
        }

        features.append(feature)

    return pd.DataFrame(features)


def train_model(df):
    df["risk_score"] = (
        df["deposits"] * 2 +
        df["repays"] * 3 -
        df["borrows"] * 1 -
        df["liquidations"] * 5
    )

    X = df[["total_tx", "unique_actions", "days_active", "tx_per_day",
            "deposits", "borrows", "repays", "redeems", "liquidations"]]
    y = df["risk_score"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    pred = model.predict(X)
    scaler = MinMaxScaler(feature_range=(0, 1000))
    df["credit_score"] = scaler.fit_transform(pred.reshape(-1, 1)).round(2)

    return df[["wallet", "credit_score"]], model


def main():
    print("Loading data...")
    with open("data/sample_transactions.json") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    print(f"Sample keys in first record: {df.iloc[0].keys()}")

    print("Engineering features...")
    features_df = engineer_features(df)

    print("Training and scoring...")
    scored_wallets, model = train_model(features_df)

    os.makedirs("outputs", exist_ok=True)
    scored_wallets.to_csv("outputs/wallet_scores.csv", index=False)
    print("Saved wallet scores to outputs/wallet_scores.csv")

    print("Generating score distribution graph...")
    plot_score_distribution(scored_wallets["credit_score"])

def plot_score_distribution(scores):
    bins = list(range(0, 1100, 100))
    plt.hist(scores, bins=bins, edgecolor="black", color="#4CAF50")
    plt.xlabel("Credit Score")
    plt.ylabel("Number of Wallets")
    plt.title("Wallet Credit Score Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/score_distribution.png")
    print("Saved score distribution plot to outputs/score_distribution.png")

if __name__ == "__main__":
    main()
