import scipy.io
import numpy as np
import mne

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
