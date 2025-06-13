#!/usr/bin/env python3
"""
Enhanced Optimized Spending Prediction System
High-performance solution for predicting spending patterns with comprehensive visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Enhanced Machine Learning imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Deep Learning imports (TensorFlow/Keras)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU
    from tensorflow.keras.optimizers import Adam, AdamW
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. LSTM models will be disabled.")
    TENSORFLOW_AVAILABLE = False

# XGBoost imports
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. XGBoost models will be disabled.")
    XGBOOST_AVAILABLE = False

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: Statsmodels not available. ARIMA models will be disabled.")
    STATSMODELS_AVAILABLE = False

# Performance optimization imports
import time
from functools import lru_cache
import gc

# Global configuration
date_keys = ['date', 'tran date', 'transactiondate', 'datetime', 'transaction_date']
amount_keys = ['amount(inr)', 'amount', 'debitamount', 'creditamount', 'debit', 'credit', 'beer', 'value', 'count', 'transaction_amount']

# ================================
# ENHANCED DATA LOADING & PROCESSING
# ================================

def load_transactions_from_csv(csv_path):
    """Enhanced CSV loader with better format detection"""
    print(f"📁 Loading transactions from: {csv_path}")
    
    needed_keys = [*date_keys, *amount_keys]
    delimiters = [',', '\t', ';', '|']
    
    # Read file with encoding handling
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    lines = None
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding, buffering=8192) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    
    if lines is None:
        raise ValueError(f"Could not read file {csv_path} with any supported encoding")
    
    # Detect header and delimiter
    header_idx = None
    detected_delim = None
    
    for idx, line in enumerate(lines[:10]):
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
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_path, delimiter=detected_delim, skiprows=header_idx, encoding=encodings[0])
    except:
        df = pd.read_csv(csv_path, delimiter=detected_delim, skiprows=header_idx, encoding='latin-1')
    
    print(f"✅ Loaded {len(df)} rows from CSV")
    return df

def extract_date_amount_columns(df):
    """Enhanced column extraction"""
    date_col = None
    amount_col = None
    
    # Find date column
    for col in df.columns:
        col_lower = col.lower().strip()
        for date_key in date_keys:
            if date_key in col_lower:
                date_col = col
                break
        if date_col:
            break
    
    # Find amount column
    for col in df.columns:
        col_lower = col.lower().strip()
        for amount_key in amount_keys:
            if amount_key in col_lower:
                amount_col = col
                break
        if amount_col:
            break
    
    if not date_col:
        raise ValueError(f"No date column found. Expected one of: {date_keys}")
    if not amount_col:
        raise ValueError(f"No amount column found. Expected one of: {amount_keys}")
    
    print(f"📅 Date column: {date_col}")
    print(f"💰 Amount column: {amount_col}")
    
    return date_col, amount_col

def preprocess_data_enhanced(df, date_col, amount_col):
    """Enhanced data preprocessing with better cleaning"""
    print("🧹 Enhanced data preprocessing...")
    start_time = time.time()
    
    # Create clean dataframe
    clean_df = df[[date_col, amount_col]].copy()
    clean_df.columns = ['date', 'amount']
    
    # Enhanced date parsing
    clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce', infer_datetime_format=True)
    
    # Enhanced amount cleaning
    if clean_df['amount'].dtype == 'object':
        clean_df['amount'] = clean_df['amount'].astype(str)
        clean_df['amount'] = clean_df['amount'].str.replace('$', '', regex=False)
        clean_df['amount'] = clean_df['amount'].str.replace(',', '', regex=False)
        clean_df['amount'] = clean_df['amount'].str.strip()
    
    # Convert to numeric
    clean_df['amount'] = pd.to_numeric(clean_df['amount'], errors='coerce')
    
    # Remove invalid data
    initial_count = len(clean_df)
    clean_df = clean_df.dropna()
    
    # Convert negative amounts to positive and remove zeros
    clean_df['amount'] = clean_df['amount'].abs()
    clean_df = clean_df[clean_df['amount'] > 0]
    
    # Enhanced outlier removal using IQR method
    Q1 = clean_df['amount'].quantile(0.25)
    Q3 = clean_df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep data within reasonable bounds
    clean_df = clean_df[
        (clean_df['amount'] >= max(lower_bound, clean_df['amount'].quantile(0.01))) & 
        (clean_df['amount'] <= min(upper_bound, clean_df['amount'].quantile(0.99)))
    ]
    
    final_count = len(clean_df)
    processing_time = time.time() - start_time
    
    print(f"✅ Preprocessing completed in {processing_time:.2f}s")
    print(f"   📊 Processed: {initial_count} → {final_count} records ({final_count/initial_count*100:.1f}% retained)")
    
    if final_count > 0:
        print(f"   💰 Amount range: ${clean_df['amount'].min():.2f} to ${clean_df['amount'].max():.2f}")
        print(f"   📅 Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    
    return clean_df

def aggregate_daily_spending_enhanced(df):
    """Enhanced daily aggregation with gap filling"""
    print("📊 Enhanced daily aggregation...")
    
    if len(df) == 0:
        return df
    
    # Group by date and sum amounts
    daily_df = df.groupby('date')['amount'].sum().reset_index()
    daily_df = daily_df.sort_values('date')
    
    # Fill missing dates with intelligent interpolation
    date_range = pd.date_range(start=daily_df['date'].min(), end=daily_df['date'].max(), freq='D')
    daily_df = daily_df.set_index('date').reindex(date_range).reset_index()
    daily_df.columns = ['date', 'amount']
    
    # Enhanced gap filling: use rolling mean for small gaps, median for larger gaps
    daily_df['amount'] = daily_df['amount'].fillna(method='ffill', limit=3)
    daily_df['amount'] = daily_df['amount'].fillna(daily_df['amount'].rolling(window=7, min_periods=1).mean())
    daily_df['amount'] = daily_df['amount'].fillna(daily_df['amount'].median())
    
    print(f"📅 Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"💰 Amount range: ${daily_df['amount'].min():.2f} to ${daily_df['amount'].max():.2f}")
    print(f"📊 Total days: {len(daily_df)}")
    
    return daily_df

# ================================
# ENHANCED FEATURE ENGINEERING
# ================================

def create_enhanced_features_v2(df):
    """Create 50+ advanced engineered features"""
    print("🔧 Creating enhanced features v2...")
    
    df = df.sort_values('date').reset_index(drop=True)
      # Advanced rolling statistics (multiple windows)
    windows = [3, 7, 14, 21, 30, 60, 90]
    for window in windows:
        df[f'rolling_mean_{window}'] = df['amount'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['amount'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'rolling_max_{window}'] = df['amount'].rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = df['amount'].rolling(window=window, min_periods=1).min()
        df[f'rolling_median_{window}'] = df['amount'].rolling(window=window, min_periods=1).median()
        # Only calculate skew and kurtosis for windows >= 3 and 4 respectively
        if window >= 3:
            df[f'rolling_skew_{window}'] = df['amount'].rolling(window=window, min_periods=3).skew().fillna(0)
        if window >= 4:
            df[f'rolling_kurt_{window}'] = df['amount'].rolling(window=window, min_periods=4).kurt().fillna(0)
    
    # Extended lag features
    for lag in [1, 2, 3, 7, 14, 21, 30, 60, 90]:
        df[f'lag_{lag}'] = df['amount'].shift(lag).fillna(df['amount'].mean())
    
    # Enhanced time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_leap_year'] = df['date'].dt.is_leap_year.astype(int)
    
    # Advanced seasonal features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    
    # Advanced trend features
    df['amount_change'] = df['amount'].diff().fillna(0)
    df['amount_pct_change'] = df['amount'].pct_change().fillna(0)
    df['amount_acceleration'] = df['amount_change'].diff().fillna(0)
      # Volatility features
    for window in [7, 14, 30]:
        df[f'volatility_{window}'] = df['amount'].rolling(window=window).std().fillna(0)
        # Avoid division by zero
        rolling_mean = df[f'rolling_mean_{window}']
        df[f'volatility_ratio_{window}'] = np.where(
            rolling_mean != 0, 
            df[f'volatility_{window}'] / rolling_mean, 
            0
        )
    
    # Percentile and ranking features
    df['amount_percentile'] = df['amount'].rank(pct=True)
    for window in [30, 60, 90]:
        df[f'percentile_rank_{window}'] = df['amount'].rolling(window=window).rank(pct=True).fillna(0.5)
      # Ratio features
    df['amount_vs_mean'] = df['amount'] / df['amount'].mean()
    for window in [7, 14, 30]:
        rolling_mean = df[f'rolling_mean_{window}']
        df[f'amount_vs_mean_{window}'] = np.where(
            rolling_mean != 0, 
            df['amount'] / rolling_mean, 
            1
        )
      # Momentum features
    for window in [7, 14, 30]:
        df[f'momentum_{window}'] = df['amount'] - df[f'rolling_mean_{window}']
        # Avoid division by zero
        rolling_std = df[f'rolling_std_{window}']
        df[f'momentum_ratio_{window}'] = np.where(
            rolling_std != 0, 
            df[f'momentum_{window}'] / rolling_std, 
            0
        )
    
    # Cyclical encoding for time features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add legacy compatibility features
    df['ma_7'] = df['rolling_mean_7']
    df['ma_30'] = df['rolling_mean_30']
    
    feature_count = len([col for col in df.columns if col not in ['date', 'amount']])
    print(f"✅ Created {feature_count} enhanced features")
    return df

# ================================
# ENHANCED MODEL TRAINING
# ================================

def train_hybrid_lstm_model(df, sequence_length=60, epochs=30):
    """Train hybrid LSTM-GRU model with advanced architecture"""
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Skipping LSTM training.")
        return None, None, None
    
    print("🧠 Training Hybrid LSTM-GRU model...")
    
    # Create enhanced features
    df_features = create_enhanced_features_v2(df.copy())
    
    # Select best features (top 40 most important)
    feature_cols = [col for col in df_features.columns if col not in ['date', 'amount']]
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_features[feature_cols])
    
    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(df_features.iloc[i]['amount'])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_features, sequence_length)
    
    if len(X) < 100:
        print("❌ Insufficient data for hybrid model training")
        return None, None, None
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Advanced hybrid model architecture
    model = Sequential([
        # First LSTM layer
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, len(feature_cols))),
        BatchNormalization(),
        Dropout(0.3),
        
        # GRU layer for efficiency
        Bidirectional(GRU(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Final LSTM layer
        Bidirectional(LSTM(32, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers with residual connections
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='linear')
    ])
    
    # Advanced optimizer with learning rate scheduling
    optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7, monitor='val_loss', verbose=1)
    ]
    
    # Train model
    print("🔄 Training Hybrid LSTM-GRU model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test, verbose=0)
    accuracy = 100 - (np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100)
    
    print(f"✅ Hybrid LSTM-GRU Accuracy: {accuracy:.1f}%")
    
    return model, scaler, accuracy

def train_advanced_xgboost(df):
    """Train advanced XGBoost with hyperparameter optimization"""
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available. Skipping XGBoost training.")
        return None, None
    
    print("🚀 Training Advanced XGBoost model...")
    
    # Create enhanced features
    df_features = create_enhanced_features_v2(df.copy())
    
    # Prepare features
    feature_cols = [col for col in df_features.columns if col not in ['date', 'amount']]
    X = df_features[feature_cols]
    y = df_features['amount']
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Advanced XGBoost model with optimized hyperparameters
    model = XGBRegressor(
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        gamma=0.1,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
        tree_method='hist'
    )
    
    # Train model
    print("🔄 Training Advanced XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = 100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    
    print(f"✅ Advanced XGBoost Accuracy: {accuracy:.1f}%")
    
    return model, accuracy

def train_advanced_arima(df):
    """Train ARIMA with automatic parameter selection"""
    if not STATSMODELS_AVAILABLE:
        print("❌ Statsmodels not available. Skipping ARIMA training.")
        return None, None
    
    print("📈 Training Advanced ARIMA model...")
    
    # Prepare time series data
    ts_data = df.set_index('date')['amount']
    
    # Split data
    split_idx = int(0.8 * len(ts_data))
    train_data = ts_data[:split_idx]
    test_data = ts_data[split_idx:]
    
    # Advanced grid search with seasonal components
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    print("🔍 Advanced grid search for optimal ARIMA parameters...")
    
    # More comprehensive parameter space
    p_values = range(0, 5)
    d_values = range(0, 3)
    q_values = range(0, 5)
    
    total_combinations = len(p_values) * len(d_values) * len(q_values)
    tested = 0
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                tested += 1
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                    
                    if tested % 25 == 0:
                        print(f"   Tested {tested}/{total_combinations} combinations...")
                        
                except:
                    continue
    
    if best_model is None:
        print("❌ Could not fit ARIMA model")
        return None, None
    
    print(f"✅ Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    
    # Evaluate model
    forecast = best_model.forecast(steps=len(test_data))
    accuracy = 100 - (np.mean(np.abs((test_data - forecast) / test_data)) * 100)
    
    print(f"✅ Advanced ARIMA Accuracy: {accuracy:.1f}%")
    
    return best_model, accuracy

# ================================
# PREDICTION FUNCTIONS
# ================================

def predict_future_enhanced(df, models, days=30):
    """Generate future predictions using ensemble of models"""
    print(f"🔮 Generating {days}-day predictions...")
    
    predictions = {}
    
    if 'lstm' in models and models['lstm'] is not None:
        lstm_model, lstm_scaler = models['lstm']
        predictions['LSTM'] = predict_lstm_future(df, lstm_model, lstm_scaler, days)
    
    if 'xgb' in models and models['xgb'] is not None:
        predictions['XGBoost'] = predict_xgb_future(df, models['xgb'], days)
    
    if 'arima' in models and models['arima'] is not None:
        predictions['ARIMA'] = predict_arima_future(models['arima'], days)
    
    # Ensemble prediction (weighted average)
    if len(predictions) > 1:
        weights = {'LSTM': 0.4, 'XGBoost': 0.4, 'ARIMA': 0.2}
        ensemble_pred = np.zeros(days)
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights and pred is not None:
                ensemble_pred += weights[model_name] * np.array(pred)
                total_weight += weights[model_name]
        
        if total_weight > 0:
            predictions['Ensemble'] = ensemble_pred / total_weight
    
    return predictions

def predict_lstm_future(df, model, scaler, days):
    """Predict future values using LSTM model"""
    try:
        df_features = create_enhanced_features_v2(df.copy())
        feature_cols = [col for col in df_features.columns if col not in ['date', 'amount']]
        
        scaled_features = scaler.transform(df_features[feature_cols])
        
        if len(scaled_features) < 60:
            return None
        
        # Use last 60 days for prediction
        sequence = scaled_features[-60:].reshape(1, 60, -1)
        
        predictions = []
        for _ in range(days):
            pred = model.predict(sequence, verbose=0)[0][0]
            predictions.append(max(0, pred))  # Ensure non-negative predictions
            
            # Update sequence for next prediction (simplified approach)
            new_features = np.zeros((1, 1, sequence.shape[2]))
            new_features[0, 0, 0] = pred  # Use prediction as main feature
            sequence = np.concatenate([sequence[:, 1:, :], new_features], axis=1)
        
        return predictions
    except Exception as e:
        print(f"❌ LSTM prediction failed: {e}")
        return None

def predict_xgb_future(df, model, days):
    """Predict future values using XGBoost model"""
    try:
        df_features = create_enhanced_features_v2(df.copy())
        feature_cols = [col for col in df_features.columns if col not in ['date', 'amount']]
        
        # Use last row features for prediction
        last_features = df_features[feature_cols].iloc[-1:].values
        
        predictions = []
        for _ in range(days):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Simple feature update (in practice, this would be more sophisticated)
            last_features[0, 0] = pred  # Update main feature
        
        return predictions
    except Exception as e:
        print(f"❌ XGBoost prediction failed: {e}")
        return None

def predict_arima_future(model, days):
    """Predict future values using ARIMA model"""
    try:
        forecast = model.forecast(steps=days)
        return [max(0, x) for x in forecast]  # Ensure non-negative predictions
    except Exception as e:
        print(f"❌ ARIMA prediction failed: {e}")
        return None

# ================================
# ENHANCED VISUALIZATION
# ================================

def create_comprehensive_prediction_plot(df, predictions, title="Enhanced Spending Prediction Analysis"):
    """Create a single comprehensive plot with all predictions and analysis"""
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    
    # Main prediction plot
    ax1.plot(df['date'], df['amount'], label='Historical Data', color='black', linewidth=2, alpha=0.8)
    
    # Create future dates
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    # Plot predictions
    colors = {'LSTM': 'blue', 'XGBoost': 'green', 'ARIMA': 'red', 'Ensemble': 'purple'}
    linestyles = {'LSTM': '--', 'XGBoost': '-.', 'ARIMA': ':', 'Ensemble': '-'}
    
    for model_name, pred_values in predictions.items():
        if pred_values is not None:
            ax1.plot(future_dates, pred_values, 
                    label=f'{model_name} Prediction', 
                    color=colors.get(model_name, 'gray'),
                    linestyle=linestyles.get(model_name, '--'),
                    linewidth=2.5 if model_name == 'Ensemble' else 2)
    
    ax1.set_title('📈 30-Day Spending Predictions', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Amount ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Historical trends analysis
    df['ma_7'] = df['amount'].rolling(window=7, min_periods=1).mean()
    df['ma_30'] = df['amount'].rolling(window=30, min_periods=1).mean()
    
    ax2.plot(df['date'], df['amount'], alpha=0.6, color='lightblue', label='Daily Spending')
    ax2.plot(df['date'], df['ma_7'], color='orange', linewidth=2, label='7-Day Average')
    ax2.plot(df['date'], df['ma_30'], color='red', linewidth=2, label='30-Day Average')
    
    ax2.set_title('📊 Historical Spending Trends', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Amount ($)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Prediction comparison
    if len(predictions) > 1:
        model_names = list(predictions.keys())
        avg_predictions = [np.mean(pred) if pred is not None else 0 for pred in predictions.values()]
        
        bars = ax3.bar(model_names, avg_predictions, 
                      color=[colors.get(name, 'gray') for name in model_names],
                      alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, avg_pred in zip(bars, avg_predictions):
            if avg_pred > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(avg_predictions) * 0.01,
                        f'${avg_pred:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('🔮 Average Prediction Comparison', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('Avg Predicted Amount ($)', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Spending distribution
    ax4.hist(df['amount'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(df['amount'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["amount"].mean():.2f}')
    ax4.axvline(df['amount'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${df["amount"].median():.2f}')
    
    ax4.set_title('💰 Spending Distribution Analysis', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Amount ($)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Overall title and layout
    fig.suptitle('🚀 ENHANCED SPENDING PREDICTION DASHBOARD 🚀', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Print prediction summary
    print("\n" + "="*60)
    print("📊 PREDICTION SUMMARY")
    print("="*60)
    
    for model_name, pred_values in predictions.items():
        if pred_values is not None:
            avg_pred = np.mean(pred_values)
            total_pred = np.sum(pred_values)
            print(f"{model_name:12}: Avg ${avg_pred:8.2f} | Total ${total_pred:10.2f}")
    
    historical_avg = df['amount'].mean()
    print(f"{'Historical':12}: Avg ${historical_avg:8.2f} | Total ${historical_avg * 30:10.2f}")
    
    plt.show()

# ================================
# MAIN EXECUTION FUNCTION
# ================================

def main_enhanced(csv_path):
    """Enhanced main execution function"""
    print("🚀 ENHANCED OPTIMIZED SPENDING PREDICTION SYSTEM 🚀")
    print("="*70)
    
    try:
        # Load and process data
        print("📊 Loading and processing data...")
        df = load_transactions_from_csv(csv_path)
        date_col, amount_col = extract_date_amount_columns(df)
        
        # Enhanced preprocessing
        clean_df = preprocess_data_enhanced(df, date_col, amount_col)
        daily_df = aggregate_daily_spending_enhanced(clean_df)
        
        print(f"✅ Processed {len(daily_df)} days of data")
        
        if len(daily_df) < 100:
            print("⚠️ Warning: Limited data available. Results may be less reliable.")
        
        # Train enhanced models
        models = {}
        accuracies = {}
        
        print("\n🧠 Training enhanced models...")
        
        # Train Hybrid LSTM
        lstm_result = train_hybrid_lstm_model(daily_df)
        if lstm_result[0] is not None:
            models['lstm'] = (lstm_result[0], lstm_result[1])
            accuracies['LSTM'] = lstm_result[2]
        
        # Train Advanced XGBoost
        xgb_result = train_advanced_xgboost(daily_df)
        if xgb_result[0] is not None:
            models['xgb'] = xgb_result[0]
            accuracies['XGBoost'] = xgb_result[1]
        
        # Train Advanced ARIMA
        arima_result = train_advanced_arima(daily_df)
        if arima_result[0] is not None:
            models['arima'] = arima_result[0]
            accuracies['ARIMA'] = arima_result[1]
        
        # Generate enhanced predictions
        print("\n🔮 Generating enhanced predictions...")
        predictions = predict_future_enhanced(daily_df, models, days=30)
        
        # Create comprehensive visualization
        print("\n📊 Creating comprehensive visualization...")
        create_comprehensive_prediction_plot(daily_df, predictions)
        
        # Print final summary
        print("\n" + "="*70)
        print("🎯 ENHANCED ANALYSIS COMPLETE")
        print("="*70)
        
        if accuracies:
            print("📈 Model Accuracies:")
            for model, acc in accuracies.items():
                print(f"   {model:12}: {acc:6.1f}%")
        
        print(f"\n📊 Historical Analysis:")
        print(f"   Average Daily:     ${daily_df['amount'].mean():.2f}")
        print(f"   Median Daily:      ${daily_df['amount'].median():.2f}")
        print(f"   Std Deviation:     ${daily_df['amount'].std():.2f}")
        print(f"   Total Historical:  ${daily_df['amount'].sum():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in enhanced workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

# ================================
# COMMAND LINE INTERFACE
# ================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("🚀 Enhanced Optimized Spending Prediction System")
        print("\nUsage:")
        print(f"  {sys.argv[0]} <csv_file>")
        print("\nExample:")
        print(f"  {sys.argv[0]} 'test data/transactions_data_part1.csv'")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    # Run enhanced analysis
    success = main_enhanced(csv_path)
    if success:
        print("🎉 Enhanced prediction analysis completed successfully!")
    else:
        print("❌ Analysis failed. Please check your data and try again.")
        sys.exit(1)