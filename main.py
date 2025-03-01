import scipy.io
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# Function to extract features from epochs
def extract_features(epochs):
    features = []
    data = epochs.get_data()
    print("Running psd_array_multitaper...")
    for epoch in data:
        psd, freqs = psd_array_multitaper(epoch, sfreq=epochs.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
        mean_power = np.mean(psd, axis=1)
        features.append(mean_power)
    return np.array(features)

# Modified evaluation for participant-level Leave-One-Out
def evaluate_leave_one_out(model, X, y, subject_ids, model_name):
    unique_subjects = np.unique(subject_ids)
    y_true = []
    y_pred = []
    y
    loo = LeaveOneOut()
    
    for train_idx, test_idx in loo.split(unique_subjects):
        # Get subject IDs for this fold
        train_subjects = unique_subjects[train_idx]
        test_subject = unique_subjects[test_idx][0]  # Only 1 test subject
        
        # Create masks
        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = (subject_ids == test_subject)
        
        # Split data
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Skaliranje podatkov
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # Učenje parametrov skaliranja samo na učnih podatkih
        X_test = scaler.transform(X_test) 

        # Train model
        model.fit(X_train, y_train)
        
        # Predict for all epochs of the test subject
        preds = model.predict(X_test)
        majority_vote = np.round(np.mean(preds)).astype(int)  # Majority voting
        
        # Get true label (assumed consistent for all epochs of the subject)
        true_label = y_test[0]
        
        y_true.append(true_label)
        y_pred.append(majority_vote)
    
    # Generate final metrics
    print(f"\nResults for {model_name} (Leave-One-Out):")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Placebo', 'Caffeine']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Placebo', 'Caffeine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name} (Leave-One-Out)')
    plt.show()

# Set log level to suppress MNE warnings
mne.set_log_level("ERROR") # Delete if you want to see errors!

# Load .mat file
mat = scipy.io.loadmat("ADD YOUR FILE PATH HERE")
alleeg = mat['ALLEEG']
print("Number of participants:", len(alleeg[0]))

# Extract data, labels, and subject IDs
all_eeg_data = []
all_labels = []
all_subject_ids = []

for subject_idx, entry in enumerate(alleeg[0]):
    eeg_data = entry['data']
    n_channels, n_trials, n_samples = eeg_data.shape
    eeg_data_reshaped = eeg_data.reshape(n_channels, n_trials * n_samples)
    all_eeg_data.append(eeg_data_reshaped)
    
    group_label = entry['group']
    labels = [group_label] * n_trials
    all_labels.extend(labels)
    all_subject_ids.extend([subject_idx] * n_trials)

# Combine data and create MNE object
eeg_data_combined = np.concatenate(all_eeg_data, axis=1)
ch_names = [f'EEG{i+1}' for i in range(n_channels)]
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
raw = mne.io.RawArray(eeg_data_combined, info)

# Filter data
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

# Create events and epochs
events = np.zeros((len(all_labels), 3), int)
events[:, 0] = np.arange(len(all_labels)) * n_samples
events[:, 2] = np.where(np.array(all_labels) == 'Caffeine', 1, 2).flatten()
event_id = {'Caffeine': 1, 'Placebo': 2}

epochs = mne.Epochs(raw_filtered, events, event_id, tmin=0, 
                    tmax=(n_samples-1)/raw.info['sfreq'], 
                    baseline=None, preload=True)

# Extract features and labels
X = extract_features(epochs)
y = np.array([1 if label == 'Caffeine' else 0 for label in all_labels])[epochs.selection]
subject_ids = np.array(all_subject_ids)[epochs.selection]

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Lahka Gradientna Boosting metoda": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    # Add more models here (Če želimo)
}

# Evaluate all models with Leave-One-Out
for model_name, model in models.items():
    (print(f"Running {model_name}..."))
    evaluate_leave_one_out(model, X, y, subject_ids, model_name)