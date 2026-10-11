import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import joblib
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import gc

# --- Configuration Dictionary ---
# Centralizes hyperparameters and file paths for easier maintenance
CONFIG = {
    'batch_size': 1024,
    'num_encoder_layers': 3,
    'dim_feedforward': 512,
    'd_k': 80,             # Dimension per attention head
    'num_heads': 8,
    'dropout': 0.1,
    'model_load_path': 'transformer_nids.pth',
    'preproc_load_path': 'preprocessors.pkl',
    'data_folder': './Dataset' 
}
CONFIG['d_model'] = CONFIG['num_heads'] * CONFIG['d_k']

# --- Model Architecture: Transformer for Network Intrusion Detection ---
class TransformerNIDS(nn.Module):
    def __init__(self, input_dim, num_classes, config):
        super(TransformerNIDS, self).__init__()
        # Projects raw features into the model's internal dimension (d_model)
        self.embedding = nn.Linear(input_dim, config['d_model'])
        
        # Standard Transformer Encoder block (Multi-head attention + Feed Forward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_encoder_layers'])
        
        # Final classification layer
        self.classifier = nn.Linear(config['d_model'], num_classes)

    def forward(self, x):
        # x shape: [batch, sequence_length, features]
        x = self.embedding(x)
        output = self.transformer_encoder(x)
        # Extract the representation of the first "token" (similar to [CLS] token in BERT)
        output = output[:, 0, :]
        output = self.classifier(output)
        return output

# --- Data Preparation Function ---
# This function replicates the training preprocessing to ensure the test set is consistent
def recover_test_split():
    print("--- Reconstructing Data for Testing ---")
    
    # 1. Load Preprocessors (Scaler and LabelEncoder used during training)
    if not os.path.exists(CONFIG['preproc_load_path']):
        raise FileNotFoundError("The preprocessors.pkl file was not found!")
        
    preproc = joblib.load(CONFIG['preproc_load_path'])
    scaler = preproc['scaler']
    le = preproc['encoder']
    saved_input_dim = preproc['input_dim']
    
    print("Preprocessors loaded.")

    # 2. Iterate through CSV files in the dataset folder
    all_files = glob.glob(os.path.join(CONFIG['data_folder'], "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV found in {CONFIG['data_folder']}")

    chunks_list = []
    
    # Non-numeric or metadata columns that shouldn't be fed to the model
    cols_to_drop = [
        'Flow ID', 'Src IP', 'Source IP', 'Dst IP', 'Destination IP', 
        'Timestamp', 'Date', 'SimillarHTTP', 'Flow Bytes/s', 'Flow Packets/s'
    ]

    print("Rereading data (identical to training)...")
    for filename in all_files:
        try:
            # Processing in chunks to manage memory efficiency
            for chunk in pd.read_csv(filename, chunksize=50000, low_memory=False):
                
                chunk.columns = chunk.columns.str.strip()
                
                # Dynamic label column identification
                if 'Label' in chunk.columns: label_col = 'Label'
                elif 'class' in chunk.columns: label_col = 'class'
                else:
                    candidates = [c for c in chunk.columns if 'Label' in c]
                    if candidates: label_col = candidates[0]
                    else: continue

                # Clean headers and drop irrelevant columns
                chunk = chunk[chunk[label_col] != label_col]
                existing_drop = [c for c in cols_to_drop if c in chunk.columns]
                chunk = chunk.drop(columns=existing_drop, errors='ignore')

                # Data Balancing: Downsample Benign traffic to prevent class imbalance
                df_benign = chunk[chunk[label_col] == 'Benign']
                df_attack = chunk[chunk[label_col] != 'Benign']
                if len(df_benign) > 0:
                    df_benign = df_benign.sample(frac=0.01, random_state=42)
                
                chunk_filtered = pd.concat([df_benign, df_attack])
                if chunk_filtered.empty: continue

                # Data Cleaning: Convert to numeric and handle Inf/NaN values
                y_chunk = chunk_filtered[label_col]
                X_chunk = chunk_filtered.drop(columns=[label_col])
                X_chunk = X_chunk.apply(pd.to_numeric, errors='coerce')
                X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)
                
                temp_df = pd.concat([X_chunk, y_chunk], axis=1).dropna()
                if temp_df.empty: continue

                X_clean = temp_df.drop(columns=[label_col]).astype(np.float32)
                y_clean = temp_df[label_col]

                chunks_list.append(pd.concat([X_clean, y_clean], axis=1))

        except Exception as e:
            pass

    if not chunks_list:
        raise ValueError("ERROR: No data recovered for testing (empty list).")

    # 3. Concatenate and finalize feature matrix X and target vector y
    df = pd.concat(chunks_list, ignore_index=True)
    del chunks_list
    gc.collect() # Manual garbage collection to free RAM
    
    print(f"Recovered raw data: {len(df)} rows.")

    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    y_str = df[label_col].astype(str).values
    X = df.drop(columns=[label_col]).values

    # Final sanitization
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print("Applying existing Scaler and Encoder...")
    
    # Filter out labels that weren't present during the training phase
    known_mask = np.isin(y_str, le.classes_)
    if not np.all(known_mask):
        print(f"Warning: {np.sum(~known_mask)} rows ignored (unknown labels).")
        X = X[known_mask]
        y_str = y_str[known_mask]

    try:
        y = le.transform(y_str)
        X = scaler.transform(X)
    except ValueError as e:
        print(f"Dimension error: The model expects {saved_input_dim} features, but we have {X.shape[1]}.")
        raise e

    # 4. Split to extract the exact same 30% Test Set as used in training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Final Test Set ready: {X_test.shape[0]} samples.")
    return X_test, y_test, saved_input_dim, len(le.classes_), le.classes_

# --- Main Execution Loop ---
if __name__ == "__main__":
    # Select hardware (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    try:
        # Load and prepare data
        X_test, y_test, input_dim, num_classes, class_names = recover_test_split()
        
        # Reshape data for Transformer: (Samples, TimeSteps=1, Features)
        X_test = X_test.reshape(-1, 1, input_dim)
        
        # Create PyTorch DataLoader
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # Initialize and load model weights
        model = TransformerNIDS(input_dim, num_classes, CONFIG).to(device)
        model.load_state_dict(torch.load(CONFIG['model_load_path'], map_location=device))
        model.eval() # Set model to evaluation mode (disables dropout)
        
        print("\n--- Starting Inference ---")
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation for faster inference and lower memory usage
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # --- Metrics and Visualization ---
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n>>> ACCURACY : {acc*100:.2f}% <<<")
        
        print("\nDetailed Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Plot Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")