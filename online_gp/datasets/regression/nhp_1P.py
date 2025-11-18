import torch
import os
from scipy.io import loadmat
from torch.utils.data import TensorDataset, random_split
from pathlib import Path
import pandas as pd
from probeinterface import Probe, read_probeinterface
from online_gp.utils.data import interpret_stim_xy
import numpy as np


class NHP_1P(object):
    def __init__(
            self, dataset_dir=None, dataset_name=None, recordings=None,
            window_name=None, probe_path=None, target_roi=None,
            subsample_ratio=1.0, test_ratio=0.1, split_seed=0,
            shuffle=True, shuffle_seed=42, get_ground_truth=True,
            **kwargs):

        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name
        self.window_name = window_name
        self.recordings = recordings
        self.probe_path = probe_path
        self.target_roi = target_roi
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.get_ground_truth = get_ground_truth

        self.shuffle_rng = np.random.default_rng(seed=self.shuffle_seed) if self.shuffle else None

        self.recording_dirs = [
            d
            for idx, d in enumerate(self.dataset_dir.iterdir())
            if d.is_dir() and (idx in self.recordings)]

        probe_group = read_probeinterface(probe_path)
        probe = probe_group.probes[0]
        self.probe = probe.to_dataframe()
        for key, value in probe.contact_annotations.items():
            self.probe.loc[:, key] = value

        self.train_dataset, self.test_dataset, self.ground_truth = self._preprocess(
            subsample_ratio, test_ratio, split_seed)

    def _preprocess(self, subsample_ratio, test_ratio, split_seed):

        target_x0, target_y0 = self.target_roi[0], self.target_roi[1]
        target_x1, target_y1 = self.target_roi[2], self.target_roi[3]

        targets_list = []
        inputs_list = []
        for recording_dir in self.recording_dirs:
            file_path = recording_dir / f'{self.dataset_name}.parquet'
            window_path = recording_dir / f'{self.window_name}.parquet'
            data = pd.read_parquet(file_path)

            window_info = pd.read_parquet(window_path).reindex(index=data.index)
            # TODO grab stim parameters from the window info and probe map
            stim_locations = pd.DataFrame(
                window_info['stim_site'].apply(interpret_stim_xy, probe=self.probe).to_list())
            inputs_list.append(
                torch.tensor(stim_locations.to_numpy(), dtype=torch.get_default_dtype())
            )
            # turn the lfp target into a scalar
            target_locations = self.probe.loc[data.columns, ['x', 'y']]
            target_mask = (
                (target_locations['x'] >= target_x0) &
                (target_locations['y'] >= target_y0) &
                (target_locations['x'] < target_x1) &
                (target_locations['y'] < target_y1)
            )
            targets_list.append(
                torch.tensor(
                    data.loc[:, target_mask].mean(axis='columns').to_numpy(),
                    dtype=torch.get_default_dtype()).unsqueeze(-1)
                )
            del data

        targets = torch.cat(targets_list, dim=0)
        inputs = torch.cat(inputs_list, dim=0)

        if self.get_ground_truth:
            input_labels = [f"input_{idx:0>2d}" for idx in range(inputs.shape[1])]
            cat_data = pd.DataFrame(
                np.concatenate([inputs.numpy(), targets.numpy()], axis=-1),
                columns=input_labels + ['target'],)
            ground_truth = cat_data.groupby(input_labels).mean().reset_index()
            # import matplotlib.pyplot as plt
            # plt.plot(cat_data['target'], label='target')
            # plt.show()
        else:
            ground_truth = None

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        input_max, _ = inputs.max(0)
        input_min, _ = inputs.min(0)
        input_range = input_max - input_min
        inputs = 2 * ((inputs - input_min) / input_range - 0.5)
        target_mean, target_std = targets.detach().mean(0), targets.detach().std(0)
        targets = (targets - target_mean) / target_std

        if self.get_ground_truth:
            ground_truth['target'] = (ground_truth['target'] - target_mean.cpu().numpy()[0]) / target_std.cpu().numpy()[0]

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
        return train_dataset, test_dataset, ground_truth
