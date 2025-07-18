
# Dual-Branch CNN–GRU–Attention Model for Asthma Risk Prediction
# Author: Vo Thanh Ha, Le An Khanh

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, Concatenate, Multiply, Softmax
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Simulated Dataset Creation ---
def create_synthetic_data(num_samples=1000, time_steps=7):
    np.random.seed(42)
    air_data = np.random.rand(num_samples, time_steps, 6)  # PM2.5, PM10, NO2, SO2, CO, O3
    weather_data = np.random.rand(num_samples, time_steps, 5)  # Temp, Humidity, Wind, Rain, Pressure
    labels = np.random.randint(0, 2, num_samples)
    return air_data, weather_data, labels

# --- Attention Mechanism ---
def attention_block(x):
    attention = Dense(x.shape[-1], activation='tanh')(x)
    attention = Dense(1, activation='softmax')(attention)
    attention_weights = Multiply()([x, attention])
    return attention_weights

# --- Build Model ---
def build_model(input_shape_air, input_shape_weather):
    input_air = Input(shape=input_shape_air)
    input_weather = Input(shape=input_shape_weather)

    x1 = Conv1D(64, kernel_size=3, activation='relu')(input_air)
    x1 = Dropout(0.3)(x1)
    x1 = attention_block(x1)
    x1 = tf.reduce_mean(x1, axis=1)

    x2 = GRU(64, return_sequences=True)(input_weather)
    x2 = Dropout(0.3)(x2)
    x2 = attention_block(x2)
    x2 = tf.reduce_mean(x2, axis=1)

    combined = Concatenate()([x1, x2])
    dense = Dense(64, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_air, input_weather], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Main ---
air_data, weather_data, labels = create_synthetic_data()
X_train_air, X_test_air, X_train_weather, X_test_weather, y_train, y_test = train_test_split(
    air_data, weather_data, labels, test_size=0.2, random_state=42)

model = build_model((7, 6), (7, 5))
model.fit([X_train_air, X_train_weather], y_train, epochs=20, batch_size=64, validation_split=0.2)

# --- Evaluation ---
y_pred = (model.predict([X_test_air, X_test_weather]) > 0.5).astype(int).flatten()
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))



# --- CNN-only Model for Ablation Study ---
def build_cnn_only_model(input_shape_air):
    input_air = Input(shape=input_shape_air)
    x = Conv1D(64, kernel_size=3, activation='relu')(input_air)
    x = Dropout(0.3)(x)
    x = attention_block(x)
    x = tf.reduce_mean(x, axis=1)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_air, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Train and Evaluate CNN-only Model ---
cnn_model = build_cnn_only_model((7, 6))
cnn_model.fit(X_train_air, y_train, epochs=20, batch_size=64, validation_split=0.2)

y_pred_cnn = (cnn_model.predict(X_test_air) > 0.5).astype(int).flatten()
print("CNN-only Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_cnn))
print("Precision:", precision_score(y_test, y_pred_cnn))
print("Recall:", recall_score(y_test, y_pred_cnn))
print("F1 Score:", f1_score(y_test, y_pred_cnn))
print("AUC:", roc_auc_score(y_test, y_pred_cnn))



# --- GRU-only Model for Ablation Study ---
def build_gru_only_model(input_shape_weather):
    input_weather = Input(shape=input_shape_weather)
    x = GRU(64, return_sequences=True)(input_weather)
    x = Dropout(0.3)(x)
    x = attention_block(x)
    x = tf.reduce_mean(x, axis=1)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_weather, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Train and Evaluate GRU-only Model ---
gru_model = build_gru_only_model((7, 5))
gru_model.fit(X_train_weather, y_train, epochs=20, batch_size=64, validation_split=0.2)

y_pred_gru = (gru_model.predict(X_test_weather) > 0.5).astype(int).flatten()
print("GRU-only Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_gru))
print("Precision:", precision_score(y_test, y_pred_gru))
print("Recall:", recall_score(y_test, y_pred_gru))
print("F1 Score:", f1_score(y_test, y_pred_gru))
print("AUC:", roc_auc_score(y_test, y_pred_gru))

# --- Baseline Models using Flattened Features ---
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

X_train_flat = np.concatenate([X_train_air.reshape(len(X_train_air), -1),
                               X_train_weather.reshape(len(X_train_weather), -1)], axis=1)
X_test_flat = np.concatenate([X_test_air.reshape(len(X_test_air), -1),
                              X_test_weather.reshape(len(X_test_weather), -1)], axis=1)

# Logistic Regression
lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
lr.fit(X_train_flat, y_train)
y_pred_lr = lr.predict(X_test_flat)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_pred_lr))

# Support Vector Machine
svm = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))])
svm.fit(X_train_flat, y_train)
y_pred_svm = svm.predict(X_test_flat)
y_pred_svm_proba = svm.predict_proba(X_test_flat)[:, 1]
print("Support Vector Machine:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))
print("AUC:", roc_auc_score(y_test, y_pred_svm_proba))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)
y_pred_rf_proba = rf.predict_proba(X_test_flat)[:, 1]
print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_pred_rf_proba))

# LSTM Model
from tensorflow.keras.layers import LSTM

def build_lstm_model(input_shape_combined):
    input_combined = Input(shape=input_shape_combined)
    x = LSTM(64, return_sequences=False)(input_combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_combined, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train_combined = np.concatenate([X_train_air, X_train_weather], axis=2)
X_test_combined = np.concatenate([X_test_air, X_test_weather], axis=2)

lstm_model = build_lstm_model((7, 11))
lstm_model.fit(X_train_combined, y_train, epochs=20, batch_size=64, validation_split=0.2)

y_pred_lstm = (lstm_model.predict(X_test_combined) > 0.5).astype(int).flatten()
print("LSTM Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print("Precision:", precision_score(y_test, y_pred_lstm))
print("Recall:", recall_score(y_test, y_pred_lstm))
print("F1 Score:", f1_score(y_test, y_pred_lstm))
print("AUC:", roc_auc_score(y_test, y_pred_lstm))
