# Import necessary libraries for deep learning and data processing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import os
import glob
import gc

# Configuration dictionary for model and training parameters
CONFIG = {
    'batch_size': 1024,
    'num_encoder_layers': 3,
    'dim_feedforward': 512,
    'd_k': 80,
    'num_heads': 8,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'epochs': 35,
    'model_save_path': 'transformer_nids.pth',
    'preproc_save_path': 'preprocessors.pkl',
    'data_folder': './Dataset'
}
CONFIG['d_model'] = CONFIG['num_heads'] * CONFIG['d_k']

# Transformer-based NIDS (Network Intrusion Detection System) model
class TransformerNIDS(nn.Module):
    def __init__(self, input_dim, num_classes, config):
        super(TransformerNIDS, self).__init__()
        self.embedding = nn.Linear(input_dim, config['d_model'])

        # Transformer encoder layer with multi-head attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_encoder_layers'])
        self.classifier = nn.Linear(config['d_model'], num_classes)

    def forward(self, x):
        # Embedding: project input to model dimension
        x = self.embedding(x)
        output = self.transformer_encoder(x)
        output = output[:, 0, :]
        output = self.classifier(output)
        return output

# Load data from CSV files, clean, preprocess, and return train/test splits
def load_and_split_data():
    print("--- Loading and Cleaning (V3 - Anti-Infinity) ---")
    all_files = glob.glob(os.path.join(CONFIG['data_folder'], "*.csv"))
    
    if not all_files:
        print(f"ERROR: No CSV found in {CONFIG['data_folder']}")
        return None, None, None, None

    chunks_list = []
    
    # Columns to remove (non-feature columns like IP addresses, timestamps, etc.)
    cols_to_drop = [
        'Flow ID', 'Src IP', 'Source IP', 'Dst IP', 'Destination IP', 
        'Timestamp', 'Date', 'SimillarHTTP', 'Flow Bytes/s', 'Flow Packets/s'
    ]

    for filename in all_files:
        print(f" -> Reading: {os.path.basename(filename)}")
        try:
            # Read CSV in chunks to handle large files memory-efficiently
            for chunk in pd.read_csv(filename, chunksize=50000, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                
                # Find and identify the label column
                if 'Label' in chunk.columns: label_col = 'Label'
                elif 'class' in chunk.columns: label_col = 'class'
                else:
                    candidates = [c for c in chunk.columns if 'Label' in c]
                    if candidates: label_col = candidates[0]
                    else: continue

                # Remove unnecessary columns
                chunk = chunk[chunk[label_col] != label_col]
                existing_drop = [c for c in cols_to_drop if c in chunk.columns]
                chunk = chunk.drop(columns=existing_drop, errors='ignore')

                # Separate benign and attack traffic
                df_benign = chunk[chunk[label_col] == 'Benign']
                df_attack = chunk[chunk[label_col] != 'Benign']
                if len(df_benign) > 0:
                    df_benign = df_benign.sample(frac=0.01, random_state=42)
                
                # Merge benign and attack data
                chunk_filtered = pd.concat([df_benign, df_attack])
                if chunk_filtered.empty: continue

                # Separate features and labels
                y_chunk = chunk_filtered[label_col]
                X_chunk = chunk_filtered.drop(columns=[label_col])

                # Convert to numeric values, coercing errors to NaN
                X_chunk = X_chunk.apply(pd.to_numeric, errors='coerce')

                # Replace infinity values with NaN for later handling
                X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)

                # Combine features and labels for cleaning
                temp_df = pd.concat([X_chunk, y_chunk], axis=1)
                # Remove rows with missing values
                temp_df = temp_df.dropna()

                if temp_df.empty: continue

                X_clean = temp_df.drop(columns=[label_col]).astype(np.float32)
                y_clean = temp_df[label_col]

                # Prepare final chunk
                chunk_final = pd.concat([X_clean, y_clean], axis=1)
                chunks_list.append(chunk_final)

        except Exception as e:
            print(f"Warning on {filename}: {e}")

    if not chunks_list:
        raise ValueError("Empty Dataset after filtering.")

    print("Merging data")
    df = pd.concat(chunks_list, ignore_index=True)
    del chunks_list
    gc.collect()

    print(f"Merged Dataset: {len(df)} rows. Final check...")

    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    y_str = df[label_col].astype(str).values
    X = df.drop(columns=[label_col]).values

    # Check and handle remaining infinite values
    if not np.isfinite(X).all():
        print("Infinite values remain. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode string labels to numeric values
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    
    # Normalize features to [0, 1] range using MinMaxScaler
    print("Starting Scaler...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Split data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    joblib.dump({'scaler': scaler, 'encoder': le, 'input_dim': X.shape[1]}, CONFIG['preproc_save_path'])
    
    return X_train, y_train, X.shape[1], len(le.classes_)

if __name__ == "__main__":
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load, preprocess, and split data
    X_train, y_train, input_dim, num_classes = load_and_split_data()
    
    # Reshape data to (batch_size, sequence_length=1, features)
    X_train = X_train.reshape(-1, 1, input_dim)
    
    # Create PyTorch dataset and dataloader
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = TransformerNIDS(input_dim, num_classes, CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("--- Starting Training ---")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    print("Model saved.")