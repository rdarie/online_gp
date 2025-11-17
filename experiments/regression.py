import hydra
import random
from omegaconf import OmegaConf, DictConfig
from upcycle.random.seed import set_all_seeds
import time
import pandas as pd
from online_gp.utils.dkl import pretrain_stem
from gpytorch.settings import *
from upcycle import cuda
import ctypes
from matplotlib import pyplot as plt
from pathlib import Path
from online_gp.settings import detach_interp_coeff

import seaborn as sns
sns.set(style='whitegrid')

def startup(hydra_cfg):
    if hydra_cfg.seed is None:
        seed = random.randint(0, 100000)
        hydra_cfg['seed'] = seed
        set_all_seeds(seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    if hydra_cfg.dtype == 'float32':
        torch.set_default_dtype(torch.float32)
    elif hydra_cfg.dtype == 'float64':
        torch.set_default_dtype(torch.float64)

    print(hydra_cfg)
    print(f"GPU available: {torch.cuda.is_available()}")

    return hydra_cfg, logger


def get_model(config, init_x, init_y, streaming):
    stem = hydra.utils.instantiate(config.stem)
    model_kwargs = dict(stem=stem, init_x=init_x, init_y=init_y)
    model = hydra.utils.instantiate(config.model, **model_kwargs)
    return cuda.try_cuda(model)


def online_regression(
        batch_model, online_model, train_x, train_y, test_x, test_y,
        update_stem, update_gp, batch_size, logger, logging_freq):

    online_rmse = online_nll = 0
    batch_rmse = batch_nll = 0
    logger.add_table('online_metrics')
    num_chunks = train_x.size(-2) // batch_size

    for t, (x, y) in enumerate(zip(train_x.chunk(num_chunks), train_y.chunk(num_chunks))):
        with detach_interp_coeff(True):
            o_rmse, o_nll = online_model.evaluate(x, y)
            # training scores on one chunk (usually 1 sample)

        start_clock = time.perf_counter()
        stem_loss, gp_loss = online_model.update(x, y, update_stem=update_stem, update_gp=update_gp)
        step_time = time.perf_counter() - start_clock

        with torch.no_grad():
            b_rmse, b_nll = batch_model.evaluate(x, y)
            # evaluate the batch model (which has already seen all of train_x, train_y) on one chunk
        online_rmse += o_rmse
        online_nll += o_nll
        batch_rmse += b_rmse
        batch_nll += b_nll

        # regret = online_rmse - batch_rmse
        # changed from original for interpretability
        regret = o_rmse - b_rmse
        num_steps = (t + 1) * batch_size

        if t % logging_freq == (logging_freq - 1):
            with detach_interp_coeff(True):
                rmse, nll = online_model.evaluate(test_x, test_y)
                print(f'T: {t+1}, test RMSE: {rmse:0.4f}, test NLL: {nll:0.4f}')
                online_hyperparams = {}
                for name, param in online_model.named_parameters():
                    online_hyperparams[f'online_{name}'] = param.detach().cpu().numpy().tolist()
            logger.log(dict(
                stem_loss=stem_loss,
                gp_loss=gp_loss,
                batch_rmse=batch_rmse,
                batch_nll=batch_nll,
                online_rmse=online_rmse,
                online_nll=online_nll,
                regret=regret,
                test_rmse=rmse,
                test_nll=nll,
                noise=online_model.noise.mean().item(),
                step_time=step_time,
                **online_hyperparams,
            ), step=num_steps, table_name='online_metrics')
            logger.write_csv()


def regression_trial(config):
    config, logger = startup(config)

    datasets = hydra.utils.instantiate(config.dataset)
    train_x, train_y = datasets.train_dataset[:]
    test_x, test_y = datasets.test_dataset[:]
    config.stem.input_dim = config.dataset.input_dim = train_x.size(-1)

    batch_model = get_model(config, train_x, train_y, streaming=False)
    # ?? the streaming parameter is not used

    if config.pretrain_stem.enabled:
        print('==== pretraining stem ====')
        loss_fn = torch.nn.MSELoss()
        batch_pretrain_stem_metrics = pretrain_stem(
            batch_model.stem, train_x, train_y, loss_fn,
            **config.pretrain_stem)
        logger.add_table('batch_pretrain_stem_metrics', batch_pretrain_stem_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['batch_pretrain_stem_metrics'])
        print(pretrain_df.tail(5).to_markdown())

    print('==== training GP in batch setting ====')
    # we train a batch_model on the entire training dataset

    batch_model.set_lr(gp_lr=config.dataset.base_lr, stem_lr=config.dataset.base_lr / 10)
    batch_metrics = batch_model.fit(train_x, train_y, config.num_batch_epochs, datasets.test_dataset)
    logger.add_table('batch_metrics', batch_metrics)
    logger.write_csv()
    batch_df = pd.DataFrame(logger.data['batch_metrics'], index=None)
    print(batch_df.tail(5).to_markdown())

    num_init_obs = int(config.model.init_ratio * train_x.size(0))
    init_x, train_x = train_x[:num_init_obs], train_x[num_init_obs:]
    init_y, train_y = train_y[:num_init_obs], train_y[num_init_obs:]
    print(f'==== training model in online setting, N: {train_x.size(0)} ====')
    online_model = get_model(config, init_x, init_y, streaming=True)

    if config.pretrain_stem.enabled:
        print('==== pretraining stem ====')
        loss_fn = torch.nn.MSELoss()
        online_pretrain_stem_metrics = pretrain_stem(
            online_model.stem, init_x, init_y, loss_fn,
            **config.pretrain_stem)
        logger.add_table('online_pretrain_stem_metrics', online_pretrain_stem_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['online_pretrain_stem_metrics'])
        print(pretrain_df.tail(5).to_markdown())

    if config.pretrain:
        # then, initially train the online_model on a subset (init_ratio, e.g. 5%) of the training dataset
        print('==== pretraining gp ====')

        online_model.set_lr(gp_lr=config.dataset.base_lr, stem_lr=config.dataset.base_lr / 10)
        pretrain_metrics = online_model.fit(init_x, init_y, config.num_batch_epochs, datasets.test_dataset)
        logger.add_table('pretrain_metrics', pretrain_metrics)
        logger.write_csv()
        pretrain_df = pd.DataFrame(logger.data['pretrain_metrics'])
        print(pretrain_df.tail(5).to_markdown())

    online_model.set_lr(gp_lr=config.dataset.base_lr / 10, stem_lr=config.dataset.base_lr / 100)
    online_regression(
        batch_model, online_model, train_x, train_y, test_x, test_y,
        config.update_stem, config.update_gp, config.batch_size, logger, config.logging_freq)
    online_df = pd.DataFrame(logger.data['online_metrics'], index=None)
    print(online_df.tail(5).to_markdown())

    # hyperparameters
    if config.make_gof_plots:
        ground_truth_map = datasets.ground_truth
        dummy_inputs = torch.tensor(
            ground_truth_map[['x', 'y']].to_numpy(),
            dtype=torch.get_default_dtype()
        )

        if torch.cuda.is_available():
            dummy_inputs = dummy_inputs.cuda()

        input_max, _ = dummy_inputs.max(0)
        input_min, _ = dummy_inputs.min(0)
        input_range = input_max - input_min
        dummy_inputs = 2 * ((dummy_inputs - input_min) / input_range - 0.5)

        o_pred_mean, o_pred_var = online_model.predict(dummy_inputs)
        online_map = ground_truth_map.copy()
        online_map['emg'] = o_pred_mean.detach().cpu().numpy()
        online_var_map = ground_truth_map.copy()
        online_var_map['emg'] = o_pred_var.detach().cpu().numpy()

        b_pred_mean, b_pred_var = batch_model.predict(dummy_inputs)
        batch_map = ground_truth_map.copy()
        batch_map['emg'] = b_pred_mean.detach().cpu().numpy()
        batch_var_map = ground_truth_map.copy()
        batch_var_map['emg'] = b_pred_var.detach().cpu().numpy()

        pivoted_gt = ground_truth_map.pivot(
            index='y', columns='x', values='emg'
        )
        pivoted_o_pred = online_map.pivot(
            index='y', columns='x', values='emg'
        )
        pivoted_b_pred = batch_map.pivot(
            index='y', columns='x', values='emg'
        )
        pivoted_o_pred_var = online_var_map.pivot(
            index='y', columns='x', values='emg'
        )
        pivoted_b_pred_var = batch_var_map.pivot(
            index='y', columns='x', values='emg'
        )

        fig, ax = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(12, 8)
        sns.heatmap(pivoted_gt, ax=ax[0, 0])
        ax[0, 0].set_title('Ground truth mean')
        sns.heatmap(pivoted_o_pred, ax=ax[0, 1])
        ax[0, 1].set_title('Online mean')
        sns.heatmap(pivoted_b_pred, ax=ax[0, 2])
        ax[0, 2].set_title('Batch mean')
        sns.heatmap(pivoted_o_pred_var, ax=ax[1, 1], cmap='crest')
        ax[1, 1].set_title('Online variance')
        sns.heatmap(pivoted_b_pred_var, ax=ax[1, 2], cmap='crest')
        ax[1, 2].set_title('Batch variance')
        fig.tight_layout()
        fig.savefig('predictions.png')

        fig, ax = plt.subplots()
        ax.plot(online_df['step'], online_df['batch_rmse'].diff(), label='Batch RMSE')
        ax.plot(online_df['step'], online_df['online_rmse'].diff(), label='Online RMSE')
        ax.set_title('step')
        ax.set_ylabel('RMSE')
        fig.tight_layout()
        fig.savefig(Path('RMSE.png'))




# Load WinMM
winmm = ctypes.WinDLL("winmm")


class TimerResolution:
    def __init__(self, ms: int = 1):
        """
        Context manager for Windows system timer resolution.
        :param ms: Desired resolution in milliseconds (commonly 1 or 0.5).
        """
        self.ms = ms
        self._active = False

    def __enter__(self):
        result = winmm.timeBeginPeriod(self.ms)
        if result != 0:
            raise OSError(f"timeBeginPeriod failed with code {result}")
        print('started high precision timer')
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._active:
            result = winmm.timeEndPeriod(self.ms)
            if result != 0:
                raise OSError(f"timeEndPeriod failed with code {result}")
            print('ended high precision timer')
            self._active = False


@hydra.main(config_path='../config', config_name='regression')
def main(config):
    with max_root_decomposition_size(config.gpytorch_global_settings.max_root_decomposition_size),\
         max_cholesky_size(config.gpytorch_global_settings.max_cholesky_size),\
         cg_tolerance(config.gpytorch_global_settings.cg_tolerance):
        regression_trial(config)


if __name__ == '__main__':
    with TimerResolution(1):   # request 1 ms system timer resolution
        print("Inside context: timer resolution set to 1 ms")
        main()
