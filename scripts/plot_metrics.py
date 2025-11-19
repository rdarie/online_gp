from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set(style='darkgrid')

base_dir_1 = Path(r'C:\Users\MBO\Documents\GitHub\online_gp\experiments\data\experiments\regression\model-dataset-version')
base_dir_2 = Path(r'C:\Users\MBO\Documents\GitHub\online_gp\data\experiments\regression\model-dataset-version')

# results_dir = base_dir_2 / r'wiski_gp_regression-3droad-0.0.13\trial_0\2025-11-11_15-01-28'

dir_list = [
    ## # no gp update
    ## base_dir_2 / r"wiski_gp_regression-neuromosaics_nhp-0.0.15\trial_0\2025-11-17_13-08-29\FlexorCarpiUlnaris",
    ## base_dir_2 / r"wiski_gp_regression-neuromosaics_nhp-0.0.15\trial_0\2025-11-17_13-08-29\ExtensorCarpiRadialis",
    ##
    ## base_dir_2 / r"wiski_gp_regression-neuromosaics_nhp-0.0.15\trial_0\2025-11-17_12-17-02\ExtensorCarpiRadialis",
    ## base_dir_2 / r"wiski_gp_regression-neuromosaics_nhp-0.0.15\trial_0\2025-11-17_12-17-02\FlexorCarpiUlnaris",
    ##
    # base_dir_2 / r"exact_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_14-05-16\ExtensorCarpiRadialis",
    # base_dir_2 / r"exact_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_14-05-16\FlexorCarpiUlnaris",
    ##
    # base_dir_2 / r"exact_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_15-25-39\ExtensorCarpiRadialis",
    # base_dir_2 / r"exact_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_15-25-39\FlexorCarpiUlnaris",
    ##
    # base_dir_2 / r"wiski_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_15-47-15",
    # base_dir_2 / r"exact_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_16-25-14",
    base_dir_2 / r"wiski_gp_regression-neuromosaics_nhp-0.0.16\trial_0\2025-11-19_16-36-08",
]

ls_col = 'online_gp.covar_module.base_kernel.base_kernel.raw_lengthscale'

for results_dir in dir_list:
    data = pd.read_csv(results_dir / 'online_metrics.csv')

    fig, ax = plt.subplots()
    ax.plot(data['step'], data['batch_rmse'].diff(), label='Batch RMSE')
    ax.plot(data['step'], data['online_rmse'].diff(), label='Online RMSE')
    ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
    ax.legend(loc='upper right')
    fig.savefig(results_dir / 'RMSE.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(data['step'], data['step_time'] * 1e3)
    ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
    ax.legend(loc='upper right')
    ax.set_xlabel('Step')
    ax.set_ylabel('Update duration (ms)')
    fig.savefig(results_dir / 'step_duration.png')
    plt.show()

    if ls_col in data.columns:
        ls = pd.DataFrame(
            data[ls_col].apply(lambda x: eval(x)[0]).to_list(),
            columns=['lengthscale00', 'lengthscale01']
        )

        fig, ax = plt.subplots()
        ax.plot(data['step'], ls['lengthscale00'], label='Online lengthscale 00')
        ax.plot(data['step'], ls['lengthscale01'], label='Online lengthscale 01')
        ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
        ax.legend(loc='upper right')
        ax.set_xlabel('Step')
        fig.savefig(results_dir / 'hyperparameters.png')
        plt.show()

# fig, ax = plt.subplots()
# ax.plot(data['step'], data['online_rmse'].diff() - data['batch_rmse'].diff(), label='Batch RMSE - Online RMSE')
# ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
# ax.legend(loc='upper right')
