import scipy.io
import numpy as np
import mne
from mne.time_frequency import psd_array_multitaper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

#* Izvlečemo značilke iz epoh
def extract_features(epochs):
    features = []
    data = epochs.get_data()  #? Izvlečemo podatke iz epoh    
    for epoch in data:
        psd, freqs = psd_array_multitaper(epoch, sfreq=epochs.info['sfreq'], fmin=1, fmax=40, n_jobs=1)
        mean_power = np.mean(psd, axis=1)
        features.append(mean_power)
    return np.array(features)

#* Naloži .mat datoteko
mat = scipy.io.loadmat('Final_ALLEEG_datasets_Coffe.mat')

#* Preverimo ključe v .mat datoteki
# print("Keys in the .mat file:", mat.keys())

#* Preverimo ključ 'ALLEEG'
alleeg = mat['ALLEEG']

#* Preverimo tip 'ALLEEG'
# print("Dtype of 'ALLEEG':", alleeg.dtype)

#* Tabele za shranjevanje podatkov
all_eeg_data = []
all_labels = []

#* Za vsak posamezen vnos v 'alleeg' izvlečemo EEG podatke
for entry in alleeg[0]:
    #* Izvlečemo EEG podatke
    eeg_data = entry['data']
    
    #* Izvlečemo obliko EEG podatkov
    # print(f"EEG data shape: {eeg_data.shape}")

    #* Izvlečemo število kanalov, število poskusov in število vzorcev
    n_channels, n_trials, n_samples = eeg_data.shape
    # print(f"Number of channels: {n_channels}, Number of trials: {n_trials}, Number of samples: {n_samples}")
    
    #! n_channels - Število EEG kanalov (elektrod).
    #! n_trials - Število poskusov (ponovitev eksperimenta).
    #! n_samples - Število časovnih točk (vzorcev), posnetih v vsakem poskusu.

    #* Spremenimo EEG podatke v 2D (n_channels, n_samples)
    eeg_data_reshaped = eeg_data.reshape(n_channels, n_trials * n_samples)

    #* Dodamo EEG podatke v tabelo
    all_eeg_data.append(eeg_data_reshaped)
    
    #* Izvlečemo oznake
    group_label = entry['group']
    #* Dodamo oznake v tabelo
    labels = [group_label] * n_trials
    all_labels.extend(labels)

#* Spremenimo tabelo v numpy array
eeg_data_combined = np.concatenate(all_eeg_data, axis=1)

#* Ustvarimo edinstvena imena kanalov
ch_names = [f'EEG{i+1}' for i in range(n_channels)]

#* Ustvarimo MNE RawArray objekt
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
raw = mne.io.RawArray(eeg_data_combined, info)

#* Shranimo RawArray objekt v datoteko
output_path = 'eeg_data_raw.fif'
raw.save(output_path, overwrite=True)

#* Filtriramo podatke
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

#* Preverimo dolžino tabele z oznakami
print("Number of labels:", len(all_labels))
print("Number of trials:", len(all_labels))

#* Ustvarimo dogodke na podlagi pravih oznak
events = np.zeros((len(all_labels), 3), int)
events[:, 0] = np.arange(len(all_labels)) * n_samples  #* Začetek dogodka
events[:, 2] = np.where(np.array(all_labels) == 'Caffeine', 1, 2).flatten()  #* Določimo ID dogodkov

#* Določimo ID dogodkov
event_id = {'Caffeine': 1, 'Placebo': 2}

#* Segmentiramo podatke v epohe
epochs = mne.Epochs(raw_filtered, events, event_id, tmin=0, tmax=(n_samples-1)/raw.info['sfreq'], baseline=None, preload=True)

#* Izpišemo število poskusov za vsako stanje
print("Number of Caffeine trials:", len(epochs['Caffeine']))
print("Number of Placebo trials:", len(epochs['Placebo']))

#* Izvlečemo značilke iz epoh
caffeine_features = extract_features(epochs['Caffeine'])
placebo_features = extract_features(epochs['Placebo'])

#* Ustvarimo oznake
caffeine_labels = np.ones(len(caffeine_features))
placebo_labels = np.zeros(len(placebo_features))

#* Združimo značilke in oznake
X = np.concatenate([caffeine_features, placebo_features], axis=0)
y = np.concatenate([caffeine_labels, placebo_labels], axis=0)

#* Razdelimo podatke na učno in testno množico
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#* Ustvarimo in naučimo model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#* Napovemo testno množico
y_pred = clf.predict(X_test)

#* Izpišemo rezultate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Placebo', 'Caffeine']))

"""
!Rezultati:
Classification Report:
              precision    recall  f1-score   support

     Placebo       1.00      1.00      1.00      4070
    Caffeine       1.00      1.00      1.00      5287

    accuracy                           1.00      9357
   macro avg       1.00      1.00      1.00      9357
weighted avg       1.00      1.00      1.00      9357

*Precision: Delež pravilno pozitivnih napovedi glede na vse pozitivne napovedi.
*Recall: Delež pravilno pozitivnih napovedi glede na vse dejansko pozitivne primere.
*F1-score: Harmonično povprečje med precision in recall.
*Support: Število primerov v vsakem razredu.
*Accuracy: Delež pravilnih napovedi glede na vse napovedi.
*Macro avg: Povprečje vseh razredov.
*Weighted avg: Povprečje vseh razredov, pri čemer so upoštevane uteži glede na število primerov v vsakem razredu.

Klasifikator je dosegel popolno uspešnost na testni množici,
z natančnostjo, odzivnostjo in F1-oceno 1,00 za obe skupini.
To nakazuje, da je klasifikator sposoben popolnoma razlikovati med skupinama:
"Placebo" in "Kofein" v tem naboru podatkov.

? 20% - testna množica | 80% - učna množica

!Celotno število coffee poskusov": 26623
!Celotno število placebo poskusov": 20160

!Število poskusov za "Caffeine" v testni množici: 26623 * 0.2 ≈ 5325
!Število poskusov za "Placebo" v testni množici: 20160 * 0.2 = 4032
"""

#* Generiramo in prikažemo matriko zmede
cm = confusion_matrix(y_test, y_pred)

#* Prikaz matrike zmede
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Placebo', 'Caffeine'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

#* Izračunamo pomembnost značilk
importances = clf.feature_importances_

#* Prikaz pomembnosti značilk
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()