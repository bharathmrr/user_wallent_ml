import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'data/sample_transactions.json'
OUTPUT_FILE = 'output/wallet_scores.csv'
PLOT_FILE = 'output/score_distribution.png'


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("Sample keys in first record:", data[0].keys())
    return pd.DataFrame(data)


def extract_amount(action_data):
    try:
        if isinstance(action_data, dict):
            for key in ['amount', 'value', 'scaledAmount']:
                if key in action_data:
                    return float(action_data[key])
    except:
        return 0.0
    return 0.0


def engineer_features(df):
    df['amount'] = df['actionData'].apply(extract_amount)

    df.rename(columns={
        'userWallet': 'user_address',
        'action': 'event_type'
    }, inplace=True)

    required_cols = ['user_address', 'event_type', 'amount']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required field '{col}' missing after renaming.")

    features = df.groupby('user_address').agg(
        total_txn=('event_type', 'count'),
        total_deposit=('amount', lambda x: x[df.loc[x.index, 'event_type'] == 'deposit'].sum()),
        total_borrow=('amount', lambda x: x[df.loc[x.index, 'event_type'] == 'borrow'].sum()),
        total_repay=('amount', lambda x: x[df.loc[x.index, 'event_type'] == 'repay'].sum()),
        total_liquidations=('event_type', lambda x: (x == 'liquidationcall').sum()),
        avg_amount=('amount', 'mean'),
        txn_types=('event_type', lambda x: len(set(x)))
    ).reset_index()

    features['repay_ratio'] = features['total_repay'] / (features['total_borrow'] + 1e-6)
    features['borrow_deposit_ratio'] = features['total_borrow'] / (features['total_deposit'] + 1e-6)

    return features.fillna(0)


def generate_scores(features_df):
    score_features = ['total_txn', 'total_deposit', 'repay_ratio',
                      'borrow_deposit_ratio', 'txn_types', 'avg_amount']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features_df[score_features])

    weights = np.array([0.15, 0.2, 0.25, -0.1, 0.2, 0.1])
    scores = np.dot(scaled, weights)
    scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()
    scores = (scores * 1000).astype(int)

    features_df['score'] = scores
    return features_df


def save_output(df):
    os.makedirs('output', exist_ok=True)
    df[['user_address', 'score']].to_csv(OUTPUT_FILE, index=False)

    bins = list(range(0, 1100, 100))
    plt.figure(figsize=(10, 6))
    plt.hist(df['score'], bins=bins, color='skyblue', edgecolor='black')
    plt.title('Wallet Score Distribution')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()


def main():
    print("Loading data...")
    df = load_data(INPUT_FILE)

    print("Engineering features...")
    features_df = engineer_features(df)

    print("Generating scores...")
    final_df = generate_scores(features_df)

    print("Saving results...")
    save_output(final_df)

    print("Done!")


if __name__ == "__main__":
    main()
