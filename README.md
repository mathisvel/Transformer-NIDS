# PA164 - MLNLP Project Implementation

Our paper was "A Transformer-based network intrusion detection approach for cloud security" 
This folder contains our implementation of what is described in the paper.

# Usage

To run the project, you have to : 

- Download the dataset : Run 'python download_datasets.py' to fetch the CSE-CIC-IDS2018 dataset
- Train the Model : Run 'python train.py' to preprocess the data and train the Transformer
- Evaluate Performance: Run 'python test.py' to generate the accuracy report and confusion matrix.

Note : In both train.py and test.py, you must update the data_folder path in the CONFIG dictionary to point to the directory where your CSV files are stored.


Below is a description of the three files used to make the project work.
The files are commented, these sections aims to save you from having to explore the code.

# download_datasets.py

This script is responsible for setting up the local environment with the necessary data:
- It downloads the "IDS 2018 Intrusion CSVs" dataset (approx. 7 GB) from Kaggle using kagglehub.
- It copies all CSV files from the cache into your local working directory.
- It performs a verification check by loading the first file (02-14-2018.csv) to ensure the download and encoding are correct.


# train.py

This script handles the data pipeline and the training of the neural network:
- Cleaning: It filters out non-numeric or metadata columns such as IP addresses, timestamps, and Flow IDs.
- Balancing: It downsamples "Benign" traffic to 1% of its original size to prevent the model from being biased toward normal traffic.
- Model: It implements a TransformerNIDS model featuring 3 encoder layers, 8 attention heads, and a classification head.
- Saving: After 35 epochs of training, it saves the model weights (transformer_nids.pth) and the preprocessors (preprocessors.pkl).

Note : expect the training to take multiple hours

# test.py

This script evaluates the trained model on unseen data:
- Consistency: It reconstructs the exact 30% test split used during the training phase by using the same random seed and stratification.
- Preprocessing: It applies the previously saved MinMaxScaler and LabelEncoder to the test data.
- Inference: It runs the test set through the model using the GPU (if available) or CPU.
- Metrics: It outputs the final accuracy score, a detailed classification report per attack type, and displays a heatmap of the confusion matrix.
