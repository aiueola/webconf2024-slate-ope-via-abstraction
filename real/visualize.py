from cycler import cycler
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import hydra
from omegaconf import DictConfig, ListConfig


cd = {
    "red": "#E24A33",
    "blue": "#348ABD",
    "purple": "#988ED5",
    "gray": "#777777",
    "green": "#8EBA42",
    "yellow": "#FBC15E",
}
colors = [cd["red"], cd["blue"], cd["purple"], cd["gray"], cd["green"], cd["yellow"]]


def visualize(
    setting: str,
    dataset: str,
    baseline_type: str,
    reward_function_types: ListConfig,
    n_random_state: int,
    log_dir: str,
):
    fig, axes = plt.subplots(1, len(reward_function_types), figsize=(7 * len(reward_function_types), 4))
    
    for i, reward_model in enumerate(reward_function_types):
        
        if setting == "slate_size":
            target_values = [4, 6, 8, 10, 12]
        elif setting == "data_size":
            target_values = [1000, 2000, 4000, 8000, 16000]

        if baseline_type == "ips":
            estimators = ["DM", "IPS", "PI", "MIPS", "LIPS (Ours)", "LIPS (w/ best β)"]
            df = pd.read_csv(f"logs/{dataset}/{reward_model}/estimation_{setting}_{n_random_state}.csv")
        elif baseline_type == "dr":
            estimators = ["DM", "DR", "PIDR", "MDR", "LIPS (Ours)", "LIPS (w/ best β)"]
            df = pd.read_csv(f"logs/{dataset}/{reward_model}/estimation_{setting}_{n_random_state}.csv")
            df_dr = pd.read_csv(f"logs/{dataset}/{reward_model}/dr_baseline/estimation_{setting}_{n_random_state}.csv")
            df[["DR", "PIDR", "MDR"]] = df_dr[["DR", "PIDR", "MDR"]]

        lips_columns = ["LIPS (0.01)", "LIPS (0.1)", "LIPS (1.0)", "LIPS (10.0)"]
        lips_all = df[lips_columns].to_numpy()

        on_policy = df["on_policy"].to_numpy()
        se_lips_all = (lips_all - on_policy[:, np.newaxis]) ** 2

        lips_best_idx = se_lips_all.argmin(axis=1)
        lips_best = lips_all[np.arange(len(lips_best_idx)), lips_best_idx]

        df["LIPS (w/ best β)"] = lips_best
        df = df[estimators + ["on_policy", "target_value", "random_state"]]

        se = (
            (df[estimators].to_numpy() - df["on_policy"].to_numpy()[:, np.newaxis]) / df["on_policy"].to_numpy()[:, np.newaxis]
        ) ** 2
        se = np.clip(se, 0, 20)

        target_value_enum = []
        for j, target_value in enumerate(target_values):
            if setting == "slate_size":
                target_value_enum.extend([target_value] * n_random_state * len(estimators))
            elif setting == "data_size":
                target_value_enum.extend([j] * n_random_state * len(estimators))

        df_for_plot = pd.DataFrame()
        df_for_plot["se"] = se.flatten()
        df_for_plot["target_value"] = target_value_enum
        df_for_plot["estimator"] = estimators * len(target_values) * n_random_state

        sns.lineplot(
            data=df_for_plot, x="target_value", y="se", hue="estimator", 
            marker="o", markersize=10, palette=colors, ax=axes[i],
        )
        
        axes[i].set_yscale("log")
                
        if setting == "slate_size":
            axes[i].set_xticks(target_values, target_values, fontsize=14)
        elif setting == "data_size":
            axes[i].set_xticks(np.arange(5), ["1000", "", "4000", "", "16000"], fontsize=14)
        
        axes[i].set_title(f"reward function ({i + 1})", fontsize=20)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("MSE (normalized)", fontsize=18)
        # axes[i].set_yscale("log")
        axes[i].legend().remove()
    
    path_ = Path(f"{log_dir}")
    path_.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_ / f"{setting}_{dataset}_{baseline_type}_mse.png", bbox_inches="tight", dpi=300)


def assert_configuration(cfg: DictConfig):
    experiment = cfg.setting.experiment
    assert experiment in ["data_size", "slate_size"]

    dataset = cfg.setting.dataset
    assert dataset in ["delicious", "wiki", "eurlex"]

    baseline_type = cfg.setting.baseline_type
    assert baseline_type in ["ips", "dr"]

    reward_function_types = cfg.setting.reward_function_types
    for reward_func in reward_function_types:
        assert reward_func in ["additive", "discount", "minmax"]

    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and n_random_state > 0


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "setting": cfg.setting.experiment,
        "dataset": cfg.setting.dataset,
        "baseline_type": cfg.setting.baseline_type,
        "reward_function_types": cfg.setting.reward_function_types,
        "n_random_state": cfg.setting.n_random_state,
        "log_dir": "logs/figs/",
    }
    # script
    visualize(**conf)


if __name__ == "__main__":
    main()
