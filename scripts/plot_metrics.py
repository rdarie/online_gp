from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set(style='darkgrid')

base_dir_1 = Path(r'C:\Users\MBO\Documents\GitHub\online_gp\experiments\data\experiments\regression\model-dataset-version')
base_dir_2 = Path(r'C:\Users\MBO\Documents\GitHub\online_gp\data\experiments\regression\model-dataset-version')

# results_dir = base_dir_2 / r'wiski_gp_regression-3droad-0.0.13\trial_0\2025-11-11_15-01-28'


dir_list = [
    ## # AdductorPollicis, grid_bound: 1
    ## base_dir_1 / r"wiski_gp_regression-neuromosaics_nhp-0.0.14\trial_0\2025-11-14_14-57-48",
    ## # AdductorPollicis, grid_bound: 0.7
    ## base_dir_1 / r"wiski_gp_regression-neuromosaics_nhp-0.0.14\trial_0\2025-11-14_15-20-05",
    ## # AdductorPollicis,
    ## base_dir_1 / r"exact_gp_regression-neuromosaics_nhp-0.0.14\trial_0\2025-11-14_15-26-35",
    ## # AdductorPollicis, grid_bound: 0.7, update_stem: false
    ## base_dir_1 / r"wiski_gp_regression-neuromosaics_nhp-0.0.14\trial_0\2025-11-14_15-37-36",
    ## FlexorCarpiUlnaris, grid_bound: 0.7, update_stem: false
    base_dir_1 / r"wiski_gp_regression-neuromosaics_nhp-0.0.14\trial_0\2025-11-14_16-01-02"
]

for results_dir in dir_list:
    data = pd.read_csv(results_dir / 'online_metrics.csv')

    fig, ax = plt.subplots()
    ax.plot(data['step'], data['batch_rmse'].diff(), label='Batch RMSE')
    ax.plot(data['step'], data['online_rmse'].diff(), label='Online RMSE')
    ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
    ax.legend(loc='upper right')
    plt.show()

# fig, ax = plt.subplots()
# ax.plot(data['step'], data['online_rmse'].diff() - data['batch_rmse'].diff(), label='Batch RMSE - Online RMSE')
# ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
# ax.legend(loc='upper right')
