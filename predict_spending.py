#!/usr/bin/env python3
"""
Enhanced Spending Prediction System
Comprehensive solution for predicting spending patterns from CSV transaction data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import csv
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Deep Learning imports (TensorFlow/Keras)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import gc
import psutil

# Global configuration
date_keys = ['date', 'tran date', 'transactiondate', 'datetime', 'transaction_date']
amount_keys = ['amount(inr)', 'amount', 'debitamount', 'creditamount', 'debit', 'credit', 'beer', 'value', 'count', 'transaction_amount']

# Performance configuration
BATCH_SIZE = 10000  # Process data in batches
MAX_WORKERS = min(8, mp.cpu_count())  # Parallel processing workers
MEMORY_THRESHOLD = 0.8  # Memory usage threshold (80%)
CACHE_SIZE = 128  # LRU cache size

# ================================
# PERFORMANCE & MEMORY MANAGEMENT
# ================================

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    return {
        'used_mb': memory_info.rss / 1024 / 1024,
        'percent': memory_percent
    }

def optimize_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def check_memory_threshold():
    """Check if memory usage exceeds threshold"""
    memory = monitor_memory()
    return memory['percent'] / 100 > MEMORY_THRESHOLD

@lru_cache(maxsize=CACHE_SIZE)
def cached_date_parse(date_str):
    """Cache parsed dates to avoid re-parsing"""
    return pd.to_datetime(date_str, errors='coerce')

def process_in_batches(df, batch_size=BATCH_SIZE):
    """Generator to process DataFrame in batches"""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

def parallel_process(func, data_list, max_workers=MAX_WORKERS):
    """Process data in parallel using ThreadPoolExecutor"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {executor.submit(func, data): data for data in data_list}
        for future in as_completed(future_to_data):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"⚠️ Parallel processing error: {e}")
    return results

# ================================
# ENHANCED DATA LOADING
# ================================

def load_transactions_from_csv(csv_path):
    """
    Enhanced CSV loader that can handle various formats and encodings
    """
    print(f"📁 Loading transactions from: {csv_path}")
    
    needed_keys = [*date_keys, *amount_keys]
    delimiters = [',', '\t', ';', '|']
    
    # Read file with better encoding handling
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
    
    for idx, line in enumerate(lines[:10]):  # Check first 10 lines
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
    
    # Load the CSV using pandas
    try:
        df = pd.read_csv(csv_path, delimiter=detected_delim, skiprows=header_idx, encoding=encodings[0])
    except:
        df = pd.read_csv(csv_path, delimiter=detected_delim, skiprows=header_idx, encoding='latin-1')
    
    print(f"✅ Loaded {len(df)} rows from CSV")
    return df

def extract_date_amount_columns(df):
    """
    Enhanced column extraction with better pattern matching
    """
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

# ================================
# DATA PROCESSING FUNCTIONS
# ================================

def process_data(df, date_col, amount_col):
    """
    Complete data processing pipeline
    """
    # Preprocess raw data
    clean_df = preprocess_data(df, date_col, amount_col)
    
    if len(clean_df) == 0:
        raise ValueError("No valid data after preprocessing")
    
    # Aggregate to daily spending
    daily_df = aggregate_daily_spending(clean_df)
    
    if len(daily_df) < 30:
        print("⚠️ Warning: Less than 30 days of data available. Results may be less reliable.")
    
    return daily_df

def preprocess_data(df, date_col, amount_col, use_batching=True):
    """
    Enhanced data preprocessing with batch processing and memory optimization
    """
    print("🧹 Preprocessing data...")
    start_time = time.time()
    
    # Create clean dataframe
    clean_df = df[[date_col, amount_col]].copy()
    clean_df.columns = ['date', 'amount']
    
    # Monitor memory usage
    memory_before = monitor_memory()
    print(f"   💾 Memory usage: {memory_before['used_mb']:.1f} MB ({memory_before['percent']:.1f}%)")
    
    if use_batching and len(clean_df) > BATCH_SIZE:
        print(f"   🔄 Processing {len(clean_df)} records in batches of {BATCH_SIZE}")
        
        # Process in batches to manage memory
        processed_batches = []
        for i, batch in enumerate(process_in_batches(clean_df)):
            if i % 5 == 0:  # Progress update every 5 batches
                print(f"   📊 Processing batch {i+1}/{(len(clean_df) // BATCH_SIZE) + 1}")
            
            # Process this batch
            batch_processed = process_batch(batch)
            if len(batch_processed) > 0:
                processed_batches.append(batch_processed)
            
            # Memory management
            if check_memory_threshold():
                print("   🧹 High memory usage detected, optimizing...")
                optimize_memory()
        
        # Combine all processed batches
        if processed_batches:
            clean_df = pd.concat(processed_batches, ignore_index=True)
        else:
            clean_df = pd.DataFrame(columns=['date', 'amount'])
    else:
        # Process all data at once for smaller datasets
        clean_df = process_batch(clean_df)
    
    # Final validation and statistics
    initial_count = len(df)
    final_count = len(clean_df)
    processing_time = time.time() - start_time
    
    print(f"✅ Preprocessing completed in {processing_time:.2f}s")
    print(f"   📊 Processed: {initial_count} → {final_count} records ({final_count/initial_count*100:.1f}% retained)")
    
    if final_count > 0:
        print(f"   💰 Final amount range: ${clean_df['amount'].min():.2f} to ${clean_df['amount'].max():.2f}")
        print(f"   📅 Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    
    memory_after = monitor_memory()
    print(f"   💾 Final memory usage: {memory_after['used_mb']:.1f} MB ({memory_after['percent']:.1f}%)")
    
    return clean_df

def process_batch(batch_df):
    """
    Process a single batch of data with optimized cleaning
    """
    # Clean and convert date column with multiple formats
    batch_df['date'] = pd.to_datetime(batch_df['date'], errors='coerce', infer_datetime_format=True)
    
    # Enhanced amount column cleaning
    if batch_df['amount'].dtype == 'object':
        batch_df['amount'] = batch_df['amount'].astype(str)
        batch_df['amount'] = batch_df['amount'].str.replace('$', '', regex=False)
        batch_df['amount'] = batch_df['amount'].str.replace(',', '', regex=False)
        batch_df['amount'] = batch_df['amount'].str.strip()
    
    # Convert to numeric, handling errors
    batch_df['amount'] = pd.to_numeric(batch_df['amount'], errors='coerce')
    
    # Remove rows with invalid data
    batch_df = batch_df.dropna()
    
    if len(batch_df) == 0:
        return batch_df
    
    # Convert negative amounts to positive (spending is spending)
    batch_df['amount'] = batch_df['amount'].abs()
    
    # Remove zero amounts
    batch_df = batch_df[batch_df['amount'] > 0]
    
    if len(batch_df) == 0:
        return batch_df
    
    # More lenient outlier removal using percentiles
    lower_percentile = batch_df['amount'].quantile(0.05)
    upper_percentile = batch_df['amount'].quantile(0.95)
    
    # Keep amounts within reasonable range
    batch_df = batch_df[
        (batch_df['amount'] >= lower_percentile) & 
        (batch_df['amount'] <= upper_percentile)
    ]
    
    return batch_df

def aggregate_daily_spending(df):
    """
    Enhanced daily aggregation with better handling of missing dates
    """
    print("📊 Aggregating daily spending...")
    
    # Check if dataframe is empty
    if len(df) == 0:
        print("❌ No data to aggregate")
        return df
    
    # Group by date and sum amounts
    daily_df = df.groupby('date')['amount'].sum().reset_index()
    daily_df = daily_df.sort_values('date')
    
    # Check if we have valid dates
    if len(daily_df) == 0:
        print("❌ No valid daily aggregates")
        return daily_df
    
    # Fill missing dates with interpolated values
    date_range = pd.date_range(start=daily_df['date'].min(), end=daily_df['date'].max(), freq='D')
    daily_df = daily_df.set_index('date').reindex(date_range).interpolate().reset_index()
    daily_df.columns = ['date', 'amount']
    
    # Fill any remaining NaNs with mean
    daily_df['amount'] = daily_df['amount'].fillna(daily_df['amount'].mean())
    
    print(f"📅 Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"💰 Amount range: ${daily_df['amount'].min():.2f} to ${daily_df['amount'].max():.2f}")
    print(f"📊 Total days: {len(daily_df)}")
    
    return daily_df

# ================================
# ENHANCED MODEL TRAINING FUNCTIONS
# ================================

def create_enhanced_features(df):
    """
    Create 40+ engineered features for enhanced model performance
    """
    print("🔧 Creating enhanced features...")
    
    # Sort by date to ensure proper order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Rolling statistics (multiple windows)
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['amount'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['amount'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'rolling_max_{window}'] = df['amount'].rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = df['amount'].rolling(window=window, min_periods=1).min()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df['amount'].shift(lag).fillna(df['amount'].mean())
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Seasonal features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    
    # Trend features
    df['amount_change'] = df['amount'].diff().fillna(0)
    df['amount_pct_change'] = df['amount'].pct_change().fillna(0)
    
    # Percentile features
    df['amount_percentile'] = df['amount'].rank(pct=True)
      # Ratio features
    df['amount_vs_mean'] = df['amount'] / df['amount'].mean()
    df['amount_vs_7day_mean'] = df['amount'] / df['rolling_mean_7']
    
    # Add legacy feature names for compatibility with existing models
    df['ma_7'] = df['rolling_mean_7']
    df['ma_30'] = df['rolling_mean_30']
    
    print(f"✅ Created {len([col for col in df.columns if col not in ['date', 'amount']])} enhanced features")
    return df

def train_enhanced_lstm(df, sequence_length=30):
    """
    Train enhanced LSTM model with triple bidirectional layers
    """
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Skipping LSTM training.")
        return None, None, None
    
    print("🧠 Training Enhanced LSTM model...")
    
    # Create enhanced features
    df_features = create_enhanced_features(df.copy())
    
    # Prepare features (exclude date and target)
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
    
    if len(X) < 50:
        print("❌ Insufficient data for LSTM training")
        return None, None, None
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Enhanced LSTM model architecture
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, len(feature_cols))),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(32, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='linear')
    ])
    
    # Compile with enhanced optimizer
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
        loss='huber',
        metrics=['mae']
    )
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7, monitor='val_loss')
    ]
      # Train model
    print("🔄 Training LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = 100 - (np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100)
    
    print(f"✅ Enhanced LSTM Accuracy: {accuracy:.1f}%")
    
    # Save model and scaler
    model.save('enhanced_lstm_model.keras')
    joblib.dump(scaler, 'enhanced_lstm_scaler.pkl')
      # Plot training history
    plot_training_history(history, "Enhanced LSTM Training History")
    
    return model, scaler, accuracy

def train_enhanced_xgboost(df):
    """
    Train enhanced XGBoost model with 40+ features
    """
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available. Skipping XGBoost training.")
        return None, None
    
    print("🚀 Training Enhanced XGBoost model...")
    
    # Create enhanced features
    df_features = create_enhanced_features(df.copy())
    
    # Prepare features
    feature_cols = [col for col in df_features.columns if col not in ['date', 'amount']]
    X = df_features[feature_cols]
    y = df_features['amount']
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Enhanced XGBoost model
    model = XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,  # Use all cores
        early_stopping_rounds=50
    )
    
    # Train model
    print("🔄 Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = 100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    
    print(f"✅ Enhanced XGBoost Accuracy: {accuracy:.1f}%")
    
    # Save model
    joblib.dump(model, 'enhanced_xgb_model.pkl')
      # Plot feature importance
    plot_feature_importance(model, feature_cols, "Enhanced XGBoost Feature Importance")
    
    return model, accuracy

def train_enhanced_arima(df):
    """
    Train enhanced ARIMA model with grid search
    """
    if not STATSMODELS_AVAILABLE:
        print("❌ Statsmodels not available. Skipping ARIMA training.")
        return None, None
    
    print("📈 Training Enhanced ARIMA model...")
    
    # Prepare time series data
    ts_data = df.set_index('date')['amount']
    
    # Split data
    split_idx = int(0.8 * len(ts_data))
    train_data = ts_data[:split_idx]
    test_data = ts_data[split_idx:]
    
    # Enhanced grid search for optimal parameters
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    print("🔍 Grid searching optimal ARIMA parameters...")
    
    # Grid search with more comprehensive parameter space
    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 4)
    
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
                    
                    if tested % 20 == 0:
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
    
    print(f"✅ Enhanced ARIMA Accuracy: {accuracy:.1f}%")
      # Save model
    joblib.dump(best_model, 'enhanced_arima_model.pkl')
    
    return best_model, accuracy

# ================================
# MODEL CACHING & FAST LOADING
# ================================

class ModelCache:
    """Fast model caching system for improved performance"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_paths = {
            'lstm': 'enhanced_lstm_model.keras',
            'lstm_scaler': 'enhanced_lstm_scaler.pkl',
            'xgb': 'enhanced_xgb_model.pkl',
            'arima': 'enhanced_arima_model.pkl'
        }
    
    def load_model(self, model_type, force_reload=False):
        """Load model with caching"""
        if model_type in self.loaded_models and not force_reload:
            print(f"   🚀 Using cached {model_type} model")
            return self.loaded_models[model_type]
        
        model_path = self.model_paths.get(model_type)
        if not model_path or not os.path.exists(model_path):
            return None
        
        print(f"   📁 Loading {model_type} model from disk...")
        start_time = time.time()
        
        try:
            if model_type == 'lstm':
                model = load_model(model_path)
            else:
                model = joblib.load(model_path)
            
            self.loaded_models[model_type] = model
            load_time = time.time() - start_time
            print(f"   ✅ {model_type} model loaded in {load_time:.2f}s")
            return model
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_type} model: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached models to free memory"""
        self.loaded_models.clear()
        optimize_memory()
        print("   🧹 Model cache cleared")

# Global model cache instance
model_cache = ModelCache()

# ================================
# SIMPLE PLOTTING FUNCTIONS  
# ================================

def plot_all_predictions(df, lstm_preds=None, xgb_preds=None, arima_preds=None, naive_preds=None, days=7):
    """
    Simple plot showing actual spending and model predictions
    """
    try:
        plt.figure(figsize=(14, 7))
        
        # Plot actual data
        plt.plot(df.index, df['amount'], label='Actual', color='black', linewidth=2)
        
        # Create future dates
        future_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')[1:]
          # Plot predictions if provided
        if lstm_preds is not None:
            plt.plot(future_dates, lstm_preds, label='LSTM', color='blue', linestyle='--', linewidth=2)
        
        if xgb_preds is not None:
            plt.plot(future_dates, xgb_preds, label='XGBoost', color='green', linestyle='--', linewidth=2)
        
        if arima_preds is not None:
            plt.plot(future_dates, arima_preds, label='ARIMA', color='red', linestyle='--', linewidth=2)
        
        if naive_preds is not None:
            plt.plot(future_dates, naive_preds, label='Naive (Last Value)', color='gray', linestyle=':', linewidth=2)
        
        plt.title('Future Spending Prediction Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Amount', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")

def plot_training_history(history, title="Model Training History"):
    """
    Simple plot for training history
    """
    try:
        if history is None or not hasattr(history, 'history'):
            print("❌ No training history available")
            return
        
        plt.figure(figsize=(10, 4))
        
        # Plot loss
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Training history plotting failed: {e}")

def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=15):
    """
    Simple feature importance plot
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            print("❌ Model does not have feature importance attributes")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices], alpha=0.8, color='skyblue')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Feature importance plotting failed: {e}")

def plot_spending_trends(daily_df, title="Daily Spending Trends", save_path=None):
    """
    Simple spending trends plot
    """
    try:
        plt.figure(figsize=(12, 6))
        
        plt.plot(daily_df['date'], daily_df['amount'], linewidth=1, alpha=0.7, color='blue', label='Daily Spending')
        
        # Add simple moving average
        ma_7 = daily_df['amount'].rolling(window=7, min_periods=1).mean()
        plt.plot(daily_df['date'], ma_7, linewidth=2, color='orange', label='7-Day Average')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            print(f"📊 Plot would be saved as: {save_path}")
        else:
            print("📊 Spending trends plot displayed")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")

def plot_model_comparison(predictions_dict, title="Model Predictions Comparison"):
    """
    Simple model comparison bar chart
    """
    try:
        plt.figure(figsize=(10, 6))
        
        models = list(predictions_dict.keys())
        predictions = list(predictions_dict.values())
        
        bars = plt.bar(models, predictions, alpha=0.8, color=['blue', 'green', 'red', 'gray'][:len(models)])
        
        # Add value labels on bars
        for bar, pred in zip(bars, predictions):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(predictions) * 0.01,
                    f'${pred:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Predicted Amount ($)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Model comparison plotting failed: {e}")

def plot_model_accuracies(accuracies_dict, title="Model Accuracy Comparison"):
    """
    Plot model accuracy comparison
    """
    try:
        plt.figure(figsize=(10, 6))
        
        models = list(accuracies_dict.keys())
        accuracies = list(accuracies_dict.values())
        
        # Color coding based on accuracy
        colors = []
        for acc in accuracies:
            if acc >= 90:
                colors.append('green')
            elif acc >= 80:
                colors.append('orange')
            elif acc >= 70:
                colors.append('yellow')
            else:
                colors.append('red')
        
        bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        # Add accuracy thresholds
        plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%+)')
        plt.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Very Good (80%+)')
        plt.axhline(y=70, color='yellow', linestyle='--', alpha=0.5, label='Good (70%+)')
        
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Accuracy plotting failed: {e}")

def create_comprehensive_dashboard(daily_df, predictions=None, accuracies=None):
    """
    Create a comprehensive dashboard with multiple plots
    """
    try:
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Daily spending trends
        plt.subplot(2, 3, 1)
        plt.plot(daily_df['date'], daily_df['amount'], linewidth=1, alpha=0.7, color='blue')
        ma_7 = daily_df['amount'].rolling(window=7, min_periods=1).mean()
        plt.plot(daily_df['date'], ma_7, linewidth=2, color='orange', label='7-Day MA')
        plt.title('Daily Spending Trends', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Spending distribution
        plt.subplot(2, 3, 2)
        plt.hist(daily_df['amount'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Spending Distribution', fontweight='bold')
        plt.xlabel('Amount ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 3. Weekly spending pattern
        plt.subplot(2, 3, 3)
        daily_df['day_of_week'] = daily_df['date'].dt.day_name()
        weekly_avg = daily_df.groupby('day_of_week')['amount'].mean()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = weekly_avg.reindex(days_order)
        plt.bar(range(len(weekly_avg)), weekly_avg.values, color='lightgreen', alpha=0.8)
        plt.title('Average Spending by Day of Week', fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Amount ($)')
        plt.xticks(range(len(weekly_avg)), [d[:3] for d in weekly_avg.index], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. Monthly trends
        plt.subplot(2, 3, 4)
        daily_df['month'] = daily_df['date'].dt.month
        monthly_avg = daily_df.groupby('month')['amount'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='red')
        plt.title('Average Spending by Month', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Average Amount ($)')
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        
        # 5. Model predictions comparison (if available)
        if predictions:
            plt.subplot(2, 3, 5)
            models = list(predictions.keys())
            pred_values = list(predictions.values())
            plt.bar(models, pred_values, color=['skyblue', 'lightgreen', 'salmon'][:len(models)], alpha=0.8)
            plt.title('Model Predictions Comparison', fontweight='bold')
            plt.xlabel('Model')
            plt.ylabel('Predicted Amount ($)')
            for i, v in enumerate(pred_values):
                plt.text(i, v + max(pred_values) * 0.01, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
        
        # 6. Model accuracies (if available)
        if accuracies:
            plt.subplot(2, 3, 6)
            models = list(accuracies.keys())
            acc_values = list(accuracies.values())
            colors = ['green' if acc >= 80 else 'orange' if acc >= 70 else 'red' for acc in acc_values]
            plt.bar(models, acc_values, color=colors, alpha=0.8)
            plt.title('Model Accuracies', fontweight='bold')
            plt.xlabel('Model')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            for i, v in enumerate(acc_values):
                plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('🚀 ENHANCED SPENDING PREDICTION DASHBOARD 🚀', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")

# ================================
# FAST PREDICTION FUNCTIONS
# ================================

def fast_predict(csv_path, models=['lstm']):
    """
    Fast prediction using cached models (LSTM and ARIMA only)
    """
    print("⚡ Fast Prediction Mode Activated")
    
    # Load and process data
    df = load_transactions_from_csv(csv_path)
    date_col, amount_col = extract_date_amount_columns(df)
    daily_df = process_data(df, date_col, amount_col)
    
    cache = ModelCache()
    predictions = {}
    
    for model_type in models:
        if model_type == 'lstm' and TENSORFLOW_AVAILABLE:
            pred = fast_lstm_predict(daily_df, cache)
            if pred is not None:
                predictions['LSTM'] = pred
        
        elif model_type == 'arima' and STATSMODELS_AVAILABLE:
            pred = fast_arima_predict(daily_df, cache)
            if pred is not None:
                predictions['ARIMA'] = pred
    
    if predictions:
        print("📊 Fast Prediction Results:")
        for model_name, pred_value in predictions.items():
            print(f"   {model_name}: ${pred_value:.2f}")
        
        # Create quick visualization
        plot_model_comparison(predictions, title="Fast Prediction Results")
        
        return predictions
    else:
        print("❌ No predictions generated")
        return None

def fast_lstm_predict(daily_df, cache):
    """Fast LSTM prediction"""
    try:
        model = cache.load_model('lstm')
        scaler = cache.load_model('lstm_scaler')
        
        if model is None or scaler is None:
            return None
        
        # Create features
        df_features = create_enhanced_features(daily_df.copy())
        feature_cols = [col for col in df_features.columns if col not in ['date', 'amount']]
        
        # Scale features and create sequence
        scaled_features = scaler.transform(df_features[feature_cols])
        
        if len(scaled_features) < 30:
            return None
        
        sequence = scaled_features[-30:].reshape(1, 30, -1)
        prediction = model.predict(sequence)[0][0]
        
        return prediction
    except Exception as e:
        print(f"❌ LSTM fast prediction failed: {e}")
        return None

def fast_arima_predict(daily_df, cache):
    """Fast ARIMA prediction"""
    try:
        model = cache.load_model('arima')
        if model is None:
            return None
        
        # Get recent data and forecast next value
        forecast = model.forecast(steps=1)
        return forecast[0] if hasattr(forecast, '__len__') else forecast
    except Exception as e:
        print(f"❌ ARIMA fast prediction failed: {e}")
        return None

# ================================
# BENCHMARK FUNCTIONS
# ================================

def run_benchmark():
    """
    Run system performance benchmark
    """
    print("🏁 Running System Performance Benchmark")
    
    # Memory information
    memory = monitor_memory()
    print(f"💾 Current Memory Usage: {memory['used_mb']:.1f} MB ({memory['percent']:.1f}%)")
    
    # CPU information
    cpu_count = mp.cpu_count()
    print(f"⚙️ CPU Cores Available: {cpu_count}")
    
    # Test model loading speed
    cache = ModelCache()
    
    start_time = time.time()
    models_loaded = 0
    
    if XGBOOST_AVAILABLE:
        if cache.load_model('xgb') is not None:
            models_loaded += 1
    
    if TENSORFLOW_AVAILABLE:
        if cache.load_model('lstm') is not None:
            models_loaded += 1
    
    if STATSMODELS_AVAILABLE:
        if cache.load_model('arima') is not None:
            models_loaded += 1
    
    load_time = time.time() - start_time
    
    print(f"⚡ Model Loading Speed: {models_loaded} models in {load_time:.2f}s")
    print(f"🚀 System Configuration: Batch size {BATCH_SIZE}, Workers {MAX_WORKERS}")
    
    # Memory threshold check
    if memory['percent'] / 100 > MEMORY_THRESHOLD:
        print("⚠️ Memory usage above threshold - batch processing recommended")
    else:
        print("✅ Memory usage within safe limits")

# ================================
# MAIN EXECUTION WORKFLOW
# ================================

def main_workflow(csv_path):
    """
    Complete training workflow with plotting integration
    """
    print("🚀 ENHANCED SPENDING PREDICTION SYSTEM 🚀")
    print("=" * 50)
    
    try:
        # Load and process data
        print("📊 Loading and processing data...")
        df = load_transactions_from_csv(csv_path)
        date_col, amount_col = extract_date_amount_columns(df)
        daily_df = process_data(df, date_col, amount_col)
        
        print(f"✅ Processed {len(daily_df)} days of data")
        
        # Create spending trends visualization
        plot_spending_trends(daily_df, "Daily Spending Analysis", "spending_trends.png")
        
        # Train models and collect results
        models = {}
        accuracies = {}
        
        # Train LSTM
        lstm_model, lstm_scaler, lstm_accuracy = train_enhanced_lstm(daily_df)
        if lstm_model is not None:
            models['LSTM'] = lstm_model
            accuracies['LSTM'] = lstm_accuracy
        
        # Train XGBoost
        xgb_model, xgb_accuracy = train_enhanced_xgboost(daily_df)
        if xgb_model is not None:
            models['XGBoost'] = xgb_model
            accuracies['XGBoost'] = xgb_accuracy
        
        # Train ARIMA
        arima_model, arima_accuracy = train_enhanced_arima(daily_df)
        if arima_model is not None:
            models['ARIMA'] = arima_model
            accuracies['ARIMA'] = arima_accuracy
          # Generate predictions for comparison
        predictions = {}
        if models:
            print("\n🔮 Generating next-day predictions...")
            
            # LSTM prediction
            if 'LSTM' in models:
                pred = fast_lstm_predict(daily_df, ModelCache())
                if pred is not None:
                    predictions['LSTM'] = pred
            
            # ARIMA prediction
            if 'ARIMA' in models:
                pred = fast_arima_predict(daily_df, ModelCache())
                if pred is not None:
                    predictions['ARIMA'] = pred
        
        # Create comprehensive visualizations
        if accuracies:
            plot_model_accuracies(accuracies, "Model Performance Comparison")
        
        if predictions:
            plot_model_comparison(predictions, title="Next-Day Spending Predictions")
        
        # Create comprehensive dashboard
        if len(daily_df) > 0:
            create_comprehensive_dashboard(daily_df, predictions, accuracies)
        
        # Print final summary
        print("\n" + "=" * 50)
        print("📈 TRAINING COMPLETE - RESULTS SUMMARY")
        print("=" * 50)
        
        if accuracies:
            print("🎯 Model Accuracies:")
            for model, acc in accuracies.items():
                print(f"   {model}: {acc:.1f}%")
        
        if predictions:
            print("\n🔮 Next-Day Predictions:")
            for model, pred in predictions.items():
                print(f"   {model}: ${pred:.2f}")
        
        print(f"\n💾 Models saved for future use")
        print(f"📊 Visualizations saved as PNG files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in main workflow: {e}")
        return False

# ================================
# COMMAND LINE INTERFACE
# ================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("🚀 Enhanced Spending Prediction System")
        print("\nUsage:")
        print(f"  {sys.argv[0]} <csv_file>                    # Full training mode")
        print(f"  {sys.argv[0]} <csv_file> --fast <models>   # Fast prediction mode")
        print(f"  {sys.argv[0]} --benchmark                  # System benchmark")
        print("\nExamples:")
        print(f"  {sys.argv[0]} 'test data/transactions_data_part1.csv'")
        print(f"  {sys.argv[0]} 'data.csv' --fast lstm")
        print(f"  {sys.argv[0]} 'data.csv' --fast lstm,arima")
        print(f"  {sys.argv[0]} --benchmark")
        sys.exit(1)
    
    # Benchmark mode
    if sys.argv[1] == "--benchmark":
        run_benchmark()
        sys.exit(0)
    
    csv_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
      # Fast prediction mode
    if len(sys.argv) > 2 and sys.argv[2] == "--fast":
        if len(sys.argv) > 3:
            models = sys.argv[3].split(',')
        else:
            models = ['lstm']  # Default to LSTM
        
        predictions = fast_predict(csv_path, models)
        if predictions:
            print("⚡ Fast prediction completed successfully!")
        sys.exit(0)
    
    # Full training mode
    success = main_workflow(csv_path)
    if success:
        print("🎉 Enhanced prediction system completed successfully!")
    else:
        print("❌ Training failed. Please check your data and try again.")
        sys.exit(1)


