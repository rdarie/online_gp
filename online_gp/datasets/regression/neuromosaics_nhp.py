import torch
import os
from scipy.io import loadmat
from torch.utils.data import TensorDataset, random_split
from pathlib import Path
import pandas as pd
from probeinterface import Probe, read_probeinterface
from online_gp.utils.data import interpret_stim_xy
import numpy as np


class Neuromosaics_NHP(object):
    def __init__(
            self, dataset_dir=None, recordings=None,
            probe_path=None, target_name=None, target_idx=None,
            subsample_ratio=1.0, test_ratio=0.1, split_seed=0,
            shuffle=True, shuffle_seed=42,
            **kwargs):

        self.dataset_dir = Path(dataset_dir)
        self.recordings = recordings
        self.probe_path = probe_path
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.shuffle_rng = np.random.default_rng(seed=self.shuffle_seed) if self.shuffle else None

        assert (target_name is not None) or (target_idx is not None)
        self.target_name = target_name
        self.target_idx = target_idx

        self.recording_paths = [
            self.dataset_dir / f"{rec}.mat"
            for rec in recordings
        ]

        # probe_group = read_probeinterface(probe_path)
        # probe = probe_group.probes[0]
        # self.probe = probe.to_dataframe()
        # for key, value in probe.contact_annotations.items():
        #     self.probe.loc[:, key] = value

        self.train_dataset, self.test_dataset = self._preprocess(
            subsample_ratio, test_ratio, split_seed)

    def _preprocess(self, subsample_ratio, test_ratio, split_seed):

        targets_list = []
        inputs_list = []
        for file_path in self.recording_paths:
            set_name = file_path.stem
            all_data = loadmat(file_path.as_posix())[set_name][0][0]

            emg_names = [f"{nn[0]}" for nn in all_data[0][0]]
            n_chan = all_data[2][0][0]

            if self.target_name is None:
                self.target_name = emg_names[self.target_idx]
            target_idx = emg_names.index(self.target_name)

            sorted_isvalid = np.concatenate(all_data[8][:, target_idx]).flatten().astype(bool)
            sorted_resp = np.concatenate(all_data[9][:, target_idx])
            sorted_resp = sorted_resp[sorted_isvalid, :]

            # sorted_resp_mean = all_data[10][:, target_idx]

            ch2xy = all_data[16]
            stim_loc_list = []
            for e_idx in range(n_chan):
                num_reps = all_data[9][e_idx, target_idx].shape[0]
                stim_loc_list.append(
                    np.tile(ch2xy[e_idx], (num_reps, 1))
                )
            stim_locations = np.concatenate(stim_loc_list)
            stim_locations = stim_locations[sorted_isvalid, :]

            del all_data

            if self.shuffle:
                perm_idx = self.shuffle_rng.permutation(stim_locations.shape[0])
                sorted_resp = sorted_resp[perm_idx, :]
                stim_locations = stim_locations[perm_idx, :]

            inputs_list.append(
                torch.tensor(stim_locations, dtype=torch.get_default_dtype())
            )
            targets_list.append(
                torch.tensor(sorted_resp[sorted_isvalid, :], dtype=torch.get_default_dtype())
                )

        targets = torch.cat(targets_list, dim=0)
        inputs = torch.cat(inputs_list, dim=0)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        input_max, _ = inputs.max(0)
        input_min, _ = inputs.min(0)
        input_range = input_max - input_min
        inputs = 2 * ((inputs - input_min) / input_range - 0.5)
        targets = (targets - targets.mean(0)) / targets.std(0)

        dataset = TensorDataset(inputs, targets)
        generator = torch.Generator().manual_seed(split_seed)
        num_samples = int(subsample_ratio * len(dataset))
        dataset, _ = random_split(
            dataset, [num_samples, len(dataset) - num_samples],
            generator=generator)
        dataset = TensorDataset(*dataset[:])

        num_test = int(test_ratio * len(dataset))
        train_dataset, test_dataset = random_split(
            dataset, [len(dataset) - num_test, num_test],
            generator=generator)
        return train_dataset, test_dataset
