import scipy.io
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Function to extract features from epochs
def extract_features(epochs):
    features = []
    data = epochs.get_data()  # Get data from epochs
    for epoch in data:
        psd, freqs = psd_array_multitaper(epoch, sfreq=epochs.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
        mean_power = np.mean(psd, axis=1)  # Mean power across frequencies
        features.append(mean_power)
    return np.array(features)

# Function to evaluate and plot results for a model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"Results for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Placebo', 'Caffeine']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Placebo', 'Caffeine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Load .mat file
mat = scipy.io.loadmat('Final_ALLEEG_datasets_Coffe.mat')
alleeg = mat['ALLEEG']
print("Number of participants:", len(alleeg[0]))  # Should print 38

# Lists to store data and labels
all_eeg_data = []
all_labels = []
all_subject_ids = []  # To store subject IDs for each epoch

# Extract EEG data, labels, and subject IDs
for subject_idx, entry in enumerate(alleeg[0]):
    eeg_data = entry['data']
    n_channels, n_trials, n_samples = eeg_data.shape
    
    # Reshape EEG data to 2D (n_channels, n_trials * n_samples)
    eeg_data_reshaped = eeg_data.reshape(n_channels, n_trials * n_samples)
    all_eeg_data.append(eeg_data_reshaped)
    
    # Extract group label (Caffeine or Placebo)
    group_label = entry['group']
    labels = [group_label] * n_trials
    all_labels.extend(labels)
    
    # Assign subject IDs for each epoch
    all_subject_ids.extend([subject_idx] * n_trials)

# Combine EEG data into a single array
eeg_data_combined = np.concatenate(all_eeg_data, axis=1)

# Create channel names
ch_names = [f'EEG{i+1}' for i in range(n_channels)]

# Create MNE RawArray object
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
raw = mne.io.RawArray(eeg_data_combined, info)

# Filter the data
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

# Create events based on labels
events = np.zeros((len(all_labels), 3), int)
events[:, 0] = np.arange(len(all_labels)) * n_samples  # Event start
events[:, 2] = np.where(np.array(all_labels) == 'Caffeine', 1, 2).flatten()  # Event IDs

# Define event IDs
event_id = {'Caffeine': 1, 'Placebo': 2}

# After creating the epochs object, check which epochs are included
epochs = mne.Epochs(raw_filtered, events, event_id, tmin=0, tmax=(n_samples-1)/raw.info['sfreq'], baseline=None, preload=True)

# Get the indices of the epochs that are included
included_epoch_indices = epochs.selection  # This gives the indices of the epochs that are included

# Filter subject_ids_combined to include only the epochs that are included
subject_ids_combined = np.array(all_subject_ids)[included_epoch_indices]

# Extract features from epochs
X = extract_features(epochs)
y = np.array([1 if label == 'Caffeine' else 0 for label in all_labels])[included_epoch_indices]  # Convert labels to binary and filter

# Debug: Check lengths
print("Length of X:", len(X))
print("Length of subject_ids_combined:", len(subject_ids_combined))
print("Number of epochs in epochs object:", len(epochs))
print("Number of included epoch indices:", len(included_epoch_indices))

# Ensure lengths match
if len(X) != len(subject_ids_combined):
    raise ValueError("Length of X and subject_ids_combined do not match. Check feature extraction and subject ID assignment.")

# Split data based on participants
unique_subjects = np.unique(subject_ids_combined)
train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)

# Create masks for training and testing sets
train_mask = np.isin(subject_ids_combined, train_subjects)
test_mask = np.isin(subject_ids_combined, test_subjects)

# Split features and labels
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Train and evaluate Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
evaluate_model(clf_rf, X_test, y_test, "Random Forest")

# Train and evaluate Logistic Regression
clf_lr = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter for convergence
clf_lr.fit(X_train, y_train)
evaluate_model(clf_lr, X_test, y_test, "Logistic Regression")

# Train and evaluate Gradient Boosting (XGBoost)
clf_xgb = XGBClassifier(n_estimators=100, random_state=42)
clf_xgb.fit(X_train, y_train)
evaluate_model(clf_xgb, X_test, y_test, "XGBoost")

# Train and evaluate k-Nearest Neighbors (k-NN)
clf_knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors by default
clf_knn.fit(X_train, y_train)
evaluate_model(clf_knn, X_test, y_test, "k-Nearest Neighbors")