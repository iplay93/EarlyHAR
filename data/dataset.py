import torch
from torch.utils import data
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data_preprocessing.dataloader import loading_data

def sliding_window_average(data, window_size, stride):
    # Calculate the number of windows resulting from the sliding process
    num_windows = (data.shape[1] - window_size) // stride + 1
    
    # Initialize an array to store the result
    result = torch.zeros((data.shape[0], num_windows, data.shape[2]))
    
    # Apply sliding window averaging
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        result[:, i, :] = torch.mean(data[:, start_idx:end_idx, :], axis=1)

    print(result.shape)    
    return result

class ExtraSensory(data.Dataset):
    def __init__(self, path_to_data):
        """
        Class for loading ExtraSensory datasets.

        Parameters
        ----------
        path_to_data : str
            Directory path containing train.pt and test.pt
        """
        super(ExtraSensory).__init__()
        X_train, y_train = torch.load(os.path.join(path_to_data, "train.pt"))
        X_test, y_test = torch.load(os.path.join(path_to_data, "test.pt"))

        self.data = X_train + X_test
        self.labels = torch.hstack([y_train, y_test]).squeeze()

        print(np.array(self.data)[0])
        print(np.array(self.labels).shape)
        print(np.array(X_train).shape)
        self.train_ix = np.arange(len(X_train))
        self.test_ix = len(X_train) + np.arange(len(X_test))

        self.data_config = {
            "N_FEATURES" : 30 if "walk" in path_to_data else 3,  # 30 features for Walking dataset, 3 for Running dataset
            "N_CLASSES" : 2,
            "nsteps" : 100,
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class DOORE(data.Dataset):
    def __init__(self, seed):
        super(DOORE).__init__()
        """
        Class for loading DOO-RE datasets.
        """
        padding, timespan, min_seq, min_samples, aug_method, aug_wise = 'mean', 10000, 10, 20, 'AddNoise', 'Temporal'

        num_classes, datalist, labellist = loading_data('doore', padding, timespan, min_seq, min_samples, aug_method, aug_wise)
        # Optionally apply sliding window averaging
        # datalist = sliding_window_average(datalist, 10, 1)

        print(datalist.shape)

        self.data = datalist
        self.labels = labellist

        # Stratified train-test split
        list_from_0_to_length = list(range(len(datalist)))
        self.train_ix, self.test_ix = train_test_split(list_from_0_to_length, test_size=0.2, stratify=labellist, random_state=seed)

        self.data_config = {
            "N_FEATURES" : 7,
            "N_CLASSES" : 4,
            "nsteps" : 598 
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class CASAS(data.Dataset):
    def __init__(self, seed):
        super(CASAS).__init__()
        """
        Class for loading CASAS datasets.
        """
        padding, timespan, min_seq, min_samples, aug_method, aug_wise = 'mean', 10000, 10, 20, 'AddNoise', 'Temporal2'

        num_classes, datalist, labellist = loading_data('casas', padding, timespan, min_seq, min_samples, aug_method, aug_wise)
        # Filter data by label condition (if needed)
        # datalist = [data for data, label in zip(datalist, labellist) if label <10]
        # labellist = [label for label in labellist if label <10]

        self.data = datalist
        self.labels = labellist

        print(self.labels)

        list_from_0_to_length = list(range(len(datalist)))
        self.train_ix, self.test_ix = train_test_split(list_from_0_to_length, test_size=0.2, stratify=labellist, random_state=seed)

        self.data_config = {
            "N_FEATURES" : 37,
            "N_CLASSES" : 14,
            "nsteps" : 46 
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class USCHAD(data.Dataset):
    def __init__(self, seed):
        super(USCHAD).__init__()
        """
        Class for loading USC-HAD datasets.
        """
        padding, timespan, min_seq, min_samples, aug_method, aug_wise = 'mean', 10, 0, 0, 'AddNoise', 'None'

        num_classes, datalist, labellist = loading_data('usc-had', padding, timespan, min_seq, min_samples, aug_method, aug_wise)
        self.data = datalist
        self.labels = labellist

        list_from_0_to_length = list(range(len(datalist)))
        self.train_ix, self.test_ix = train_test_split(list_from_0_to_length, test_size=0.2, stratify=labellist, random_state=seed)

        self.data_config = {
            "N_FEATURES" : 6,
            "N_CLASSES" : 12,
            "nsteps" : 46 
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class OpenPack(data.Dataset):
    def __init__(self, seed):
        super(OpenPack).__init__()
        """
        Class for loading OpenPack datasets.
        """
        padding, timespan, min_seq, min_samples, aug_method, aug_wise = 'mean', 100, 10, 20, 'AddNoise', 'None'

        num_classes, datalist, labellist = loading_data('openpack', padding, timespan, min_seq, min_samples, aug_method, aug_wise)
        self.data = datalist
        self.labels = labellist

        list_from_0_to_length = list(range(len(datalist)))
        self.train_ix, self.test_ix = train_test_split(list_from_0_to_length, test_size=0.2, stratify=labellist, random_state=seed)

        self.data_config = {
            "N_FEATURES" : 40,
            "N_CLASSES" : 10,
            "nsteps" : 78 
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class Opportunity(data.Dataset):
    def __init__(self, seed):
        super(Opportunity).__init__()
        """
        Class for loading Opportunity datasets.
        """
        padding, timespan, min_seq, min_samples, aug_method, aug_wise = 'mean', 1000, 10, 20, 'AddNoise', 'Temporal'

        num_classes, datalist, labellist = loading_data('opportunity', padding, timespan, min_seq, min_samples, aug_method, aug_wise)
        self.data = datalist
        self.labels = labellist

        list_from_0_to_length = list(range(len(datalist)))
        self.train_ix, self.test_ix = train_test_split(list_from_0_to_length, test_size=0.2, stratify=labellist, random_state=seed)

        self.data_config = {
            "N_FEATURES" : 242,
            "N_CLASSES" : 5,
            "nsteps" : 169
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class ARAS(data.Dataset):
    def __init__(self, seed):
        super(ARAS).__init__()
        """
        Class for loading ARAS datasets.
        """
        padding, timespan, min_seq, min_samples, aug_method, aug_wise = 'mean', 1000, 10, 20, 'AddNoise', 'Temporal'

        num_classes, datalist, labellist = loading_data('aras_a', padding, timespan, min_seq, min_samples, aug_method, aug_wise)
        self.data = datalist
        self.labels = labellist

        list_from_0_to_length = list(range(len(datalist)))
        self.train_ix, self.test_ix = train_test_split(list_from_0_to_length, test_size=0.2, stratify=labellist, random_state=seed)

        self.data_config = {
            "N_FEATURES" : 20,
            "N_CLASSES" : 22,
            "nsteps" : 63
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class SyntheticData(data.Dataset):
    def __init__(self, N=10000, T=10, mode="early_late"):
        """
        Class for generating synthetic irregularly-sampled time series data.

        Parameters
        ----------
        N : int
            Number of time series instances (classes are balanced).
        T : int
            Number of observations per time series.
        mode : str
            Distribution of signal locations: 'early', 'late', 'early_late', or 'uniform'.
        """
        super(SyntheticData, self).__init__()
        self.T = T 
        self.N = N
        if mode == "early":
            self.signal_times = np.random.normal(.25, .1, (N, 1)).clip(0, 1)
        elif mode == "late":
            self.signal_times = np.random.normal(.75, .1, (N, 1)).clip(0, 1)
        elif mode == "uniform":
            self.signal_times = np.random.uniform(0.0, 1.0, (int(N/2), 1))
        elif mode == "early_late":
            early = np.random.normal(.25, .1, (int(N//2), 1)).clip(0, 1)
            late = np.random.normal(.75, .1, (N//2, 1)).clip(0, 1)
            self.signal_times = np.concatenate((early, late))
        self.signal_times = self.signal_times[np.random.choice(self.N, self.N, replace=False)]
        self._N_FEATURES = 1
        self.data, self.labels = self.loadData()

        self.train_ix = np.random.choice(N, N, replace=False)
        self.test_ix = self.train_ix[int(0.8*N):]
        self.train_ix = self.train_ix[:int(0.8*N)]

        self.data_config = {
            "N_FEATURES" : 1,
            "N_CLASSES" : 2,
            "nsteps" : T 
        }

    def __getitem__(self, ix): 
        return self.data[ix], self.labels[ix]

    def getVals(self, timesteps, values, mask, nsteps):
        """
        Resample values, masks, and time gaps into fixed-length sequences.
        """
        V = values.shape[1]
        new_vals = np.zeros((nsteps, V))
        new_masks = np.zeros((nsteps, V))  # 1 means observed
        past = np.ones((nsteps, V))        # Time since last observation
        future = np.ones((nsteps, V))      # Time until next observation
        bins = np.round(np.linspace(0+(1./nsteps), 1, nsteps), 3)
        for v in range(V):
            t0 = timesteps[mask[:, v] == 1].numpy()
            v0 = values[mask[:, v] == 1, v].numpy()
            buckets = (np.abs(t0 - (bins[:, None]-(1./(2*nsteps))))).argmin(0)
            for n in range(nsteps):
                ix = np.where(buckets == n)
                new_vals[n, v] = np.nanmean(np.take(v0, ix))
                new_masks[n, v] = len(ix[0]) > 0
                below = t0[np.where(t0 <= bins[n])]
                if len(below) > 0:
                    past[n, v] = bins[n] - below.max()
                above = t0[np.where(t0 > bins[n])]
                if len(above) > 0:
                    future[n, v] = above.min() - bins[n]
        new_vals[np.isnan(new_vals)] = 0.0
        return new_vals.astype(np.float32), new_masks.astype(np.float32), past.astype(np.float32)

    def loadData(self):
        """
        Generate synthetic time series and labels, apply resampling.
        """
        timesteps = np.random.uniform(0, 1, (self.N, self.T-1))
        all_timesteps = np.concatenate((timesteps, self.signal_times), axis=1)

        values = np.zeros_like(timesteps)
        signals = np.concatenate((np.ones((int(self.N/2), 1)), -1*np.ones((int(self.N/2), 1))), axis=0)
        all_values = np.concatenate((values, signals), axis=1)

        sorted_ix = np.argsort(all_timesteps)
        timesteps = np.array([all_timesteps[i][sorted_ix[i]] for i in range(self.N)])
        values = np.array([all_values[i][sorted_ix[i]] for i in range(self.N)])

        labels = np.array([0]*int(self.N/2) + [1]*int(self.N/2))
        timesteps = torch.tensor(timesteps).float()
        values = torch.tensor(values).unsqueeze(2).float()

        ix = np.random.choice(self.N, self.N, replace=False)
        self.signal_times = self.signal_times[ix]
        timesteps = timesteps[ix]
        values = values[ix]
        labels = labels[ix]
        masks = torch.ones_like(values).float()

        data = [] 
        for i in range(len(values)):
            new_vals, new_masks, past = self.getVals(timesteps[i], values[i], masks[i], nsteps=self.T)
            data.append((new_vals, new_masks, past))
        labels = torch.tensor(labels, dtype=torch.long)
        return data, labels
