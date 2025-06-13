import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from xgboost import XGBRegressor
import joblib
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping

date_keys = [
    'date', 'tran date', 'transactiondate', 'datetime'
]
amount_keys = [
    'amount(inr)', 'amount', 'debitamount', 'creditamount', 'debit', 'credit', 'beer', 'value', 'count'
]

def load_transactions_from_csv(csv_path):
    import csv
    needed_keys = [*date_keys, *amount_keys]
    delimiters = [',', '\t', ';', '|']
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    header_idx = None
    detected_delim = None
    for idx, line in enumerate(lines):
        for delim in delimiters:
            cols = [c.strip().replace('"', '') for c in line.strip().split(delim)]
            for key in needed_keys:
                for col in cols:
                    if key.lower() in col.lower():
                        header_idx = idx
                        detected_delim = delim
                        break
                if header_idx is not None:
                    break
            if header_idx is not None:
                break
        if header_idx is not None:
            break
    if header_idx is None:
        raise ValueError("No valid header row found in CSV. Please check the file format.")
    import pandas as pd
    try:
        preview = pd.read_csv(csv_path, nrows=10, low_memory=False, sep=detected_delim, skiprows=header_idx)
        if len(preview.columns) == 1:
            print(f"[WARN] Only one column detected, trying to auto-detect delimiter...")
            preview = pd.read_csv(csv_path, nrows=10, low_memory=False, sep=None, engine='python', skiprows=header_idx)
        possible_cols = set(preview.columns)
        needed_keys_lower = [key.lower().strip() for key in needed_keys]
        needed_cols = [
            c for c in possible_cols if any(key in c.lower().strip() for key in needed_keys_lower)
        ]
        usecols = needed_cols if needed_cols else None
    except Exception:
        usecols = None
    chunk_size = 100_000
    records = []
    try:
        for chunk in pd.read_csv(csv_path, usecols=usecols, low_memory=False, chunksize=chunk_size, sep=detected_delim, skiprows=header_idx):
            records.extend(chunk.to_dict(orient='records'))
    except Exception:
        for chunk in pd.read_csv(csv_path, usecols=usecols, low_memory=False, chunksize=chunk_size, sep=None, engine='python', skiprows=header_idx):
            records.extend(chunk.to_dict(orient='records'))
    return records

def xlsx_to_csv(xlsx_path):
    import pandas as pd
    csv_path = os.path.splitext(xlsx_path)[0] + ".csv"
    df = pd.read_excel(xlsx_path)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Converted {xlsx_path} to {csv_path}")
    return csv_path

def load_transactions(path):
    if path.lower().endswith('.xlsx'):
        path = xlsx_to_csv(path)
    if path.lower().endswith('.csv'):
        print(f"Loading transactions from CSV: {path}")
        return load_transactions_from_csv(path)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

def safe_float(val):
    try:
        if val in [None, '', '-']:
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def preprocess_transactions(rows):
    date_keys_lower = [k.lower().strip() for k in date_keys]
    amount_keys_lower = [k.lower().strip() for k in amount_keys]
    drcr_keys = [k.lower().strip() for k in ['DR/CR', 'DrCr', 'DR|CR']]
    summary_keywords = [
        'opening balance', 'closing balance', 'total', 'summary', 'balance', 'inflow', 'outflow', 'net inflow', 'net outflow', 'receipts', 'payments', 'profit', 'payables', 'accruals', 'add:', 'less:'
    ]
    records = []
    if rows:
        print(f"[DEBUG] First row keys: {list(rows[0].keys())}")
        print("[DEBUG] First 3 rows:")
        for r in rows[:3]:
            print(r)
    for row in rows:
        stripped_row = {k.lower().strip() if isinstance(k, str) else k: v for k, v in row.items()}
        for key in stripped_row:
            val = str(stripped_row[key]).lower()
            if any(kw in val for kw in summary_keywords):
                break
        else:
            date_val = None
            for k in date_keys_lower:
                if k in stripped_row:
                    date_val = stripped_row[k]
                    break
            amount_val = None
            amount_col_used = None
            for k in amount_keys_lower:
                if k in stripped_row:
                    amount_val = stripped_row[k]
                    amount_col_used = k
                    break
            if not date_val or not amount_val:
                print(f"[SKIP] Row missing date or amount: {row}")
                continue
            if amount_val is None:
                non_date_cols = [k for k in stripped_row if k not in date_keys]
                if len(non_date_cols) == 0:
                    print(f"[ERROR] Only date column present, no amount column to process. Row: {row}")
                    continue
                elif len(non_date_cols) == 1:
                    amount_col_used = non_date_cols[0]
                    amount_val = stripped_row[amount_col_used]
                    print(f"[WARN] No known amount column found, using fallback column '{amount_col_used}' as amount.")
                else:
                    print(f"[ERROR] No known amount column found. Available columns: {list(stripped_row.keys())}")
            drcr_val = None
            for k in drcr_keys:
                if k in stripped_row:
                    drcr_val = stripped_row[k]
                    break
            print(f"Row: {row}\n  date_val: {date_val}, amount_val: {amount_val}, drcr_val: {drcr_val}, amount_col_used: {amount_col_used}")
            if amount_val is not None and drcr_val is not None:
                try:
                    amount = float(str(amount_val).replace(',', '').replace(' ', ''))
                    if str(drcr_val).strip().upper().startswith('D'):
                        amount = -abs(amount)
                    elif str(drcr_val).strip().upper().startswith('C'):
                        amount = abs(amount)
                    print(f"  Using DR/CR logic: amount={amount}")
                except Exception as e:
                    print(f"  DR/CR logic failed: {e}")
                    continue
            elif ('DebitAmount' in stripped_row or 'CreditAmount' in stripped_row) or ('Debit' in stripped_row or 'Credit' in stripped_row):
                debit = safe_float(stripped_row.get('DebitAmount', stripped_row.get('Debit', 0)))
                credit = safe_float(stripped_row.get('CreditAmount', stripped_row.get('Credit', 0)))
                amount = credit - debit
                print(f"  Using Debit/Credit logic: debit={debit}, credit={credit}, amount={amount}")
            else:
                try:
                    cleaned_amount = str(amount_val).replace('$', '').replace(',', '').strip()
                    amount = float(cleaned_amount)
                    print(f"  Using fallback logic: amount={amount}")
                except Exception as e:
                    print(f"  Fallback logic failed: {e}")
                    continue
            try:
                date = pd.to_datetime(date_val, errors='coerce')
                if pd.isnull(date):
                    try:
                        date = datetime.strptime(str(date_val).strip(), "%d-%b-%y")
                        print(f"  Parsed with custom format: {date}")
                    except Exception as e2:
                        print(f"  Date parsing failed for value '{date_val}': {e2}")
                        continue
            except Exception as e:
                print(f"  Date parsing failed: {e}")
                continue
            if pd.isnull(date):
                print(f"  Skipping row due to null date.")
                continue
            print(f"  Appending: date={date}, amount={amount}")
            records.append({'date': date, 'amount': amount})
    if not records:
        if len(rows) > 0:
            print("\n[DEBUG] No valid transactions found. First row keys:", list(rows[0].keys()))
            print("[DEBUG] All unique keys in data:", set(k for row in rows for k in row.keys()))
        raise ValueError("No valid transactions found. Check if the date and amount columns are correctly detected and parsed.")
    df = pd.DataFrame(records)
    if 'date' not in df.columns:
        raise KeyError("'date' column not found in parsed transactions. Check your input data and parsing logic.")
    df = df.sort_values('date')
    df = df.set_index('date')
    df = df.groupby(df.index).sum()
    print(f"Number of unique dates after aggregation: {len(df)}")
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def train_lstm_on_spending(df, seq_length=20, epochs=20, return_model=True):
    if len(df) <= 2:
        raise ValueError(f"Too few unique dates ({len(df)}) for LSTM training. At least 3 are required.")
    if seq_length >= len(df):
        seq_length = max(1, len(df) - 1)
        print(f"[INFO] Sequence length adjusted to {seq_length} due to limited unique dates.")
    scaler = MinMaxScaler()
    amounts = df['amount'].values.reshape(-1, 1)
    scaled = scaler.fit_transform(amounts)
    X, y = create_sequences(scaled, seq_length)
    if X.shape[0] == 0:
        raise ValueError(f"Not enough data to create sequences: only {len(df)} unique dates, but seq_length={seq_length}.")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, verbose=1, callbacks=[early_stop])
    if return_model:
        return model, scaler, X, y, seq_length
    else:
        return scaler, X, y, seq_length

def naive_baseline_predictions(df, days=30, method="last"):
    y = df['amount'].values
    if method == "last":
        return np.full(days, y[-1])
    elif method == "mean":
        mean_val = np.mean(y[-days:]) if len(y) >= days else np.mean(y)
        return np.full(days, mean_val)
    else:
        raise ValueError("Unknown method for naive baseline.")

def plot_all_predictions(df, lstm_preds, xgb_preds, arima_preds, naive_preds, days=7):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['amount'], label='Actual', color='black')
    future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
    plt.plot(future_dates, lstm_preds, label='LSTM', color='blue', linestyle='--')
    plt.plot(future_dates, xgb_preds, label='XGBoost', color='green', linestyle='--')
    plt.plot(future_dates, arima_preds, label='ARIMA', color='red', linestyle='--')
    plt.plot(future_dates, naive_preds, label='Naive (Last Value)', color='gray', linestyle=':')
    plt.title('Future Spending Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_future_spending(df, model_path='spending_lstm_model.keras', seq_length=30, days=30):
    from tensorflow.keras.models import load_model
    scaler = MinMaxScaler()
    amounts = df['amount'].values.reshape(-1, 1)
    scaled = scaler.fit_transform(amounts)
    model = load_model(model_path)
    last_seq = scaled[-seq_length:].reshape(1, seq_length, 1)
    preds = []
    for _ in range(days):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_inv

def evaluate_lstm_accuracy(df, seq_length=30, epochs=20, test_ratio=0.2):
    n = len(df)
    n_test = max(1, int(n * test_ratio))
    df_train = df.iloc[:-n_test]
    df_test = df.iloc[-(n_test+seq_length):]
    model, scaler, X_train, y_train, seq_length = train_lstm_on_spending(df_train, seq_length=seq_length, epochs=epochs)
    amounts_test = df_test['amount'].values.reshape(-1, 1)
    scaled_test = scaler.transform(amounts_test)
    X_test, y_test = create_sequences(scaled_test, seq_length)
    if X_test.shape[0] == 0:
        print("[WARN] Not enough test data for evaluation.")
        return None
    y_pred = model.predict(X_test).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    return rmse, mae

def train_xgboost_on_spending(df, seq_length=30, epochs=20, model_path='spending_xgb_model.h5'):
    y = df['amount'].values
    X = []
    for i in range(seq_length, len(y)):
        X.append(y[i-seq_length:i])
    X = np.array(X)
    y_supervised = y[seq_length:]
    if len(X) == 0:
        raise ValueError("Not enough data for XGBoost training.")
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y_supervised)
    joblib.dump(model, model_path)
    print(f"XGBoost model saved as {model_path}")
    return model, seq_length

def predict_future_xgboost(df, model_path='spending_xgb_model.h5', seq_length=30, days=7):
    model = joblib.load(model_path)
    y = df['amount'].values
    last_seq = y[-seq_length:]
    preds = []
    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, -1))[0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    return np.array(preds)

def train_arima_on_spending(df, order=(5,1,0), model_path='spending_arima_model.h5'):
    y = df['amount'].values
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    joblib.dump(model_fit, model_path)
    print(f"ARIMA model saved as {model_path}")
    return model_fit

def predict_future_arima(df, model_path='spending_arima_model.h5', days=7):
    model_fit = joblib.load(model_path)
    forecast = model_fit.forecast(steps=days)
    return forecast

def main(path):
    rows = load_transactions(path)
    df = preprocess_transactions(rows)
    print(f"Loaded {len(df)} transactions for training.")
    seq_length = 30
    epochs = 20
    print(f"\nEvaluating model accuracy (seq_length={seq_length}, epochs={epochs})...")
    acc = evaluate_lstm_accuracy(df, seq_length=seq_length, epochs=epochs)
    if acc is not None:
        rmse, mae = acc
        y_true_mean = np.mean(np.abs(df['amount']))
        if y_true_mean > 0:
            accuracy_pct = 100 - (mae / y_true_mean) * 100
            print(f"LSTM Validation Accuracy: {accuracy_pct:.2f}% (RMSE: {rmse:.4f}, MAE: {mae:.4f})")
        else:
            print("LSTM Validation Accuracy: N/A (mean true value is zero)")
    from tensorflow.keras.models import load_model
    model_path = 'spending_lstm_model.keras'
    print(f"\nTraining model with seq_length={seq_length}, epochs={epochs}")
    model, scaler, X, y, seq_length = train_lstm_on_spending(df, seq_length=seq_length, epochs=epochs, existing_model=None)
    model.save(model_path)
    print(f"Model saved as {model_path}")
    print("\nLoading saved model and making future predictions...")
    lstm_preds = predict_future_spending(df, model_path=model_path, seq_length=seq_length, days=30)
    print(f"Next 30 predicted spending values (LSTM): {lstm_preds}")
    print("\nTraining XGBoost model...")
    xgb_model, xgb_seq_length = train_xgboost_on_spending(df, seq_length=seq_length)
    xgb_preds = predict_future_xgboost(df, seq_length=xgb_seq_length, days=30)
    print(f"Next 30 predicted spending values (XGBoost): {xgb_preds}")
    print("\nTraining ARIMA model...")
    arima_model = train_arima_on_spending(df)
    arima_preds = predict_future_arima(df, days=30)
    print(f"Next 30 predicted spending values (ARIMA): {arima_preds}")
    naive_preds = naive_baseline_predictions(df, days=30, method="last")
    if len(df) > seq_length + 30:
        y_true = df['amount'].values[-30:]
        lstm_mae = np.mean(np.abs(lstm_preds - y_true))
        lstm_acc = 100 - (lstm_mae / np.mean(np.abs(y_true))) * 100 if np.mean(np.abs(y_true)) > 0 else float('nan')
        xgb_mae = np.mean(np.abs(xgb_preds - y_true))
        xgb_acc = 100 - (xgb_mae / np.mean(np.abs(y_true))) * 100 if np.mean(np.abs(y_true)) > 0 else float('nan')
        arima_mae = np.mean(np.abs(arima_preds - y_true))
        arima_acc = 100 - (arima_mae / np.mean(np.abs(y_true))) * 100 if np.mean(np.abs(y_true)) > 0 else float('nan')
        naive_mae = np.mean(np.abs(naive_preds - y_true))
        naive_acc = 100 - (naive_mae / np.mean(np.abs(y_true))) * 100 if np.mean(np.abs(y_true)) > 0 else float('nan')
        print(f"\nLSTM 30-day Accuracy: {lstm_acc:.2f}% (MAE: {lstm_mae:.4f})")
        print(f"XGBoost 30-day Accuracy: {xgb_acc:.2f}% (MAE: {xgb_mae:.4f})")
        print(f"ARIMA 30-day Accuracy: {arima_acc:.2f}% (MAE: {arima_mae:.4f})")
        print(f"Naive (Last Value) 30-day Accuracy: {naive_acc:.2f}% (MAE: {naive_mae:.4f})")
    else:
        print("Not enough data for 30-day accuracy comparison.")
    plot_all_predictions(df, lstm_preds, xgb_preds, arima_preds, naive_preds, days=30)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_spending.py <path_to_csv>")
    else:
        main(sys.argv[1])


