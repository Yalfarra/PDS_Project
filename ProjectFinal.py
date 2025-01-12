import pandas as pd
import numpy as np
import requests
import threading
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder, Binarizer, MinMaxScaler, StandardScaler, normalize


def clean_and_preprocess_data(df):
    if 'Type' in df.columns:
        label_encoder = LabelEncoder()
        df['Type'] = label_encoder.fit_transform(df['Type'])

    if 'Machine failure' in df.columns:
        binarizer = Binarizer(threshold=0.5)
        df['Machine failure binary'] = binarizer.fit_transform(df[['Machine failure']])

    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_mm'] = mm_scaler.fit_transform(df[[col]])

    s_scaler = StandardScaler()
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_sc'] = s_scaler.fit_transform(df[[col]])

    if 'Tool wear [min]' in df.columns:
        df['Tool wear normalized_l1'] = normalize(df[['Tool wear [min]']], norm="l1", axis=0)
        df['Tool wear normalized_l2'] = normalize(df[['Tool wear [min]']], norm="l2", axis=0)

    return df

if __name__ == "__main__":
    dataset_path = r"data\ai4i2020.csv"  
    try:
        maintenance_df = pd.read_csv(dataset_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Dataset file not found. Please ensure the file exists at the specified path.")
        exit()

    processed_df = clean_and_preprocess_data(maintenance_df)

    processed_df.to_csv("maintenance_processed.csv", index=False)
    print("Processed data saved to 'maintenance_processed.csv'")

    print(processed_df.head())
