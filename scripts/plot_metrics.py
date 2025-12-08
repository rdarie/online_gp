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
    base_dir_2 / r"wiski_gp_regression-NHP_NP-0.0.17\trial_0\2025-11-20_17-34-53",
]

ls_col = 'online_gp.covar_module.base_kernel.base_kernel.raw_lengthscale'
# ls_col = 'online_gp.covar_module.base_kernel.raw_lengthscale'

for results_dir in dir_list:
    data = pd.read_csv(results_dir / 'online_metrics.csv')
    data['step_time'] *= 1e3

    fig, ax = plt.subplots()
    ax.plot(data['step'], data['batch_rmse'].diff(), label='Batch RMSE')
    ax.plot(data['step'], data['online_rmse'].diff(), label='Online RMSE')
    ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
    ax.legend(loc='upper right')
    fig.savefig(results_dir / 'RMSE.png')
    plt.show()

    fig, ax = plt.subplots(
        ncols=2, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.01},
        sharey=True,
    )
    ax[0].plot(data['step'], data['step_time'])
    # ax[0].legend(loc='upper right')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Update duration (ms)')
    sns.histplot(data, y='step_time', ax=ax[1], stat='proportion')
    ax[1].set_ylabel('')
    twax = ax[1].twiny()
    sns.ecdfplot(data, y='step_time', ax=twax, stat='proportion', color='orange')
    fig.suptitle(f"{results_dir.relative_to(results_dir.parents[2])}")
    fig.savefig(results_dir / 'step_duration.png')
    plt.show()

    if ls_col in data.columns:
        ls = pd.DataFrame(
            data[ls_col].apply(lambda x: eval(x)[0]).to_list(),
        )
        ls.columns = [f"length_scale_{xx:0>2d}" for xx in range(ls.shape[1])]
        fig, ax = plt.subplots()
        for xx in range(ls.shape[1]):
            ax.plot(data['step'], ls[f"length_scale_{xx:0>2d}"], label=f'Online lengthscale {xx:0>2d}')
        ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
        ax.legend(loc='upper right')
        ax.set_xlabel('Step')
        fig.savefig(results_dir / 'hyperparameters.png')
        plt.show()

# fig, ax = plt.subplots()
# ax.plot(data['step'], data['online_rmse'].diff() - data['batch_rmse'].diff(), label='Batch RMSE - Online RMSE')
# ax.set_title(f"{results_dir.relative_to(results_dir.parents[2])}")
# ax.legend(loc='upper right')
