import os
import time
import pickle
from pathlib import Path
from typing import Union, Dict, Any
from multiprocessing import Pool
from collections import defaultdict
from copy import deepcopy

import hydra
from omegaconf import DictConfig, ListConfig

import numpy as np
import pandas as pd

from src.estimators import DirectMethod as DM
from src.estimators import DoublyRobust as DR
from src.estimators import PseudoInverseDoublyRobust as PIDR
from src.estimators import LatentDoublyRobust as LDR
from src.estimators import MarginalizedDR as MDR
from src.meta import OffPolicyEvaluation
from src.learners import LatentRepresentationLearning
from src.learners import RewardLearning
from src.utils import format_runtime

from real.dataset import SemiSyntheticSlateBanditDataset
from real.dataset import (
    clipped_reward_function,
    minmax_reward_function,
    discretized_reward_function,
    logistic_reward_function,
    additive_reward_function,
    product_reward_function,
    sparse_reward_function,
    discount_reward_function,
)


def generate_and_obtain_dataset(
    dataset: str,
    n_rounds: int,
    n_unique_action: int,
    slate_size: int,
    behavior_tau: float,
    behavior_epsilon: float,
    evaluation_epsilon: float,
    reward_type: str,
    reward_function_type: str,
    reward_std: float,
    device: str,
    random_state: int,
    base_random_state: int,
    use_base_random_state: bool,
    log_dir: str,
):
    base_random_state = base_random_state if use_base_random_state else random_state

    path_ = Path(log_dir + f"/dataset/{dataset}_{reward_type}_{reward_function_type}")
    path_.mkdir(exist_ok=True, parents=True)

    path_dataset = Path(
        path_
        / f"dataset_{n_unique_action}_{slate_size}_{base_random_state}"
    )
    path_bandit_feedback = Path(
        path_
        / f"bandit_feedback_{n_rounds}_{n_unique_action}_{slate_size}_{behavior_tau}_{behavior_epsilon}_{reward_std}_{base_random_state}_{random_state}"
    )
    path_evaluation_dict = Path(
        path_
        / f"evaluation_dict_{n_rounds}_{n_unique_action}_{slate_size}_{behavior_tau}_{behavior_epsilon}_{evaluation_epsilon}_{reward_std}_{base_random_state}_{random_state}"
    )

    if path_evaluation_dict.exists():
        with open(path_evaluation_dict, "rb") as f:
            evaluation_dict = pickle.load(f)
        with open(path_bandit_feedback, "rb") as f:
            bandit_feedback = pickle.load(f)
        return bandit_feedback, evaluation_dict

    if path_bandit_feedback.exists():
        with open(path_bandit_feedback, "rb") as f:
            bandit_feedback = pickle.load(f)
        with open(path_dataset, "rb") as f:
            dataset = pickle.load(f)

    else:
        if reward_function_type == "clip":
            reward_function = clipped_reward_function
        elif reward_function_type == "minmax":
            reward_function = minmax_reward_function
        elif reward_function_type == "discretize":
            reward_function = discretized_reward_function
        elif reward_function_type == "logistic":
            reward_function = logistic_reward_function
        elif reward_function_type == "additive":
            reward_function = additive_reward_function
        elif reward_function_type == "product":
            reward_function = product_reward_function
        elif reward_function_type == "sparse":
            reward_function = sparse_reward_function
        elif reward_function_type == "discount":
            reward_function = discount_reward_function

        if path_dataset.exists():
            with open(path_dataset, "rb") as f:
                dataset = pickle.load(f)

            dataset.reward_type = reward_type
            dataset.reward_function = reward_function
            dataset.tau = behavior_tau
            dataset.epsilon = behavior_epsilon
            dataset.reward_std = reward_std

        else:
            dataset = SemiSyntheticSlateBanditDataset(
                slate_size=slate_size,
                n_unique_action=np.full((slate_size,), n_unique_action),
                reward_type=reward_type,
                reward_function=reward_function,
                reward_std=reward_std,
                tau=behavior_tau,
                epsilon=behavior_epsilon,
                base_random_state=base_random_state,
                device=device,
                dataset_name=dataset,
            )

            with open(path_dataset, "wb") as f:
                pickle.dump(dataset, f)

        dataset.set_random_state(random_state)
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds)

    policy_logit_ = dataset.obtain_policy_logit(
        context=bandit_feedback["context"],
    )

    evaluation_dict = {}
    evaluation_dict["evaluation_policy_slot_prob_all"] = dataset.calc_pscore_all(
        policy_logit_=policy_logit_,
        epsilon=evaluation_epsilon,
    )
    evaluation_dict[
        "evaluation_policy_prob_for_chosen_action"
    ] = dataset.calc_pscore_given_action(
        action=bandit_feedback["action"],
        policy_logit_=policy_logit_,
        epsilon=evaluation_epsilon,
    )
    evaluation_dict["on_policy_policy_value"] = dataset.calc_on_policy_policy_value(
        label=bandit_feedback["label"],
        policy_logit_=policy_logit_,
        epsilon=evaluation_epsilon,
    )

    return bandit_feedback, evaluation_dict


def evaluate_estimators(
    dataset: str,
    n_rounds: int,
    n_unique_action: int,
    slate_size: int,
    behavior_tau: float,
    behavior_epsilon: float,
    evaluation_epsilon: float,
    reward_type: str,
    reward_function_type: str,
    reward_std: float,
    target_param: str,
    target_value: Union[int, float],
    batch_size: int,
    learning_rate: float,
    abstraction_type: str,
    abstraction_dimension: int,
    betas: np.ndarray,
    report_results_for_all_beta: bool,
    device: str,
    random_state: int,
    base_random_state: int,
    use_base_random_state: bool,
    log_dir: str,
    **kwargs,
):
    start = time.time()
    print(f"random_state={random_state}, {target_param}={target_value} started")
    # logged dataset
    bandit_feedback, evaluation_dict = generate_and_obtain_dataset(
        dataset=dataset,
        n_rounds=n_rounds,
        n_unique_action=n_unique_action,
        slate_size=slate_size,
        behavior_tau=behavior_tau,
        behavior_epsilon=behavior_epsilon,
        evaluation_epsilon=evaluation_epsilon,
        reward_type=reward_type,
        reward_function_type=reward_function_type,
        reward_std=reward_std,
        device=device,
        random_state=random_state,
        base_random_state=base_random_state,
        use_base_random_state=use_base_random_state,
        log_dir=log_dir,
    )
    # estimator setting
    dm = DM(estimator_name="DM")
    dr = DR(estimator_name="DR")
    pidr = PIDR(estimator_name="PIDR")
    mdr = MDR(estimator_name="MDR")
    ldr = LDR(estimator_name="LDR (Ours)")
    ope_estimators = [dm, dr, pidr, mdr, ldr]
    # models
    abstraction_learner = LatentRepresentationLearning(
        dim_context=bandit_feedback["context"].shape[1],
        n_unique_actions=np.full((slate_size,), n_unique_action),
        reward_type=reward_type,
        abstraction_type=abstraction_type,
        n_latent_abstraction=abstraction_dimension,
        dim_latent_abstraction=abstraction_dimension,
        device=device,
        random_state=base_random_state,
        log_dir=log_dir,
    )
    reward_learner = RewardLearning(
        dim_context=bandit_feedback["context"].shape[1],
        n_unique_actions=np.full((slate_size,), n_unique_action),
        reward_type=reward_type,
        device=device,
        random_state=base_random_state,
        log_dir=log_dir,
    )
    # meta class to handle ope
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=ope_estimators,
        abstraction_learner=abstraction_learner,
        reward_learner=reward_learner,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    # obtain estimation result of baseline estimators
    baseline_estimation_dict = ope.estimate_policy_value_with_baseline_estimators(
        estimator_names=["DM", "DR", "PIDR", "MDR"],
        evaluation_policy_prob_for_chosen_action=evaluation_dict[
            "evaluation_policy_prob_for_chosen_action"
        ],
        evaluation_policy_slot_prob_all=evaluation_dict[
            "evaluation_policy_slot_prob_all"
        ],
    )
    baseline_estimation_dict["on_policy"] = evaluation_dict["on_policy_policy_value"]
    # obtain estimation result of lips (our proposal)
    lips_estimation_dict = ope.estimate_policy_value_with_lips(
        estimator_name="LDR (Ours)",
        evaluation_policy_slot_prob_all=evaluation_dict[
            "evaluation_policy_slot_prob_all"
        ],
        betas=betas,
        report_results_for_all_beta=report_results_for_all_beta,
        require_reward_prediction=True,
    )

    finish = time.time()
    print(
        f"random_state={random_state}, {target_param}={target_value} finished",
        format_runtime(start, finish),
    )
    return baseline_estimation_dict, lips_estimation_dict


def assert_configuration(cfg: DictConfig):
    experiment = cfg.setting.experiment
    assert experiment in ["data_size", "slate_size", "default", "ablation"]

    dataset = cfg.setting.dataset
    assert dataset in ["delicious", "bibtex", "wiki", "eurlex"]

    start_random_state = cfg.setting.start_random_state
    assert isinstance(start_random_state, int) and start_random_state >= 0

    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and n_random_state > start_random_state

    base_random_state = cfg.setting.base_random_state
    assert isinstance(base_random_state, int) and base_random_state > 0

    n_unique_action = cfg.setting.n_unique_action
    assert isinstance(n_unique_action, int) and n_unique_action > 0

    n_rounds = cfg.setting.n_rounds
    if isinstance(n_rounds, ListConfig):
        for value in n_rounds:
            assert isinstance(value, int) and value > 0
    else:
        assert isinstance(n_rounds, int) and n_rounds > 0

    slate_size = cfg.setting.slate_size
    if isinstance(slate_size, ListConfig):
        for value in slate_size:
            assert isinstance(value, int) and value > 0
    else:
        assert isinstance(slate_size, int) and slate_size > 0

    behavior_tau = cfg.setting.behavior_tau
    assert isinstance(behavior_tau, float) and behavior_tau != 0.0

    behavior_epsilon = cfg.setting.behavior_epsilon
    assert isinstance(behavior_epsilon, float) and 0.0 <= behavior_epsilon <= 1.0

    evaluation_epsilon = cfg.setting.evaluation_epsilon
    assert isinstance(evaluation_epsilon, float) and 0.0 <= evaluation_epsilon <= 1.0

    reward_type = cfg.setting.reward_type
    assert reward_type in ["binary", "continuous"]

    reward_function_type = cfg.setting.reward_function_type
    assert reward_function_type in [
        "clip",
        "minmax",
        "discretize",
        "logistic",
        "additive",
        "product",
        "sparse",
        "discount",
    ]

    reward_std = cfg.setting.reward_std
    assert isinstance(reward_std, float) and reward_std >= 0.0

    target_param = cfg.setting.target_param
    assert target_param in [
        "n_rounds",
        "slate_size",
        "None",
    ]

    batch_size = cfg.model_params.batch_size
    assert isinstance(batch_size, int) and batch_size > 0

    learning_rate = cfg.model_params.learning_rate
    assert isinstance(learning_rate, float) and learning_rate > 0

    abstraction_type = cfg.model_params.abstraction_type
    assert abstraction_type in ["discrete", "continuous"]

    abstraction_dimension = cfg.model_params.abstraction_dimension
    assert isinstance(abstraction_dimension, int) and abstraction_dimension > 0

    beta = cfg.model_params.beta
    for value in beta:
        assert isinstance(value, float) and value >= 0.0

    device = cfg.model_params.device
    assert device in ["cpu", "cuda", "mps"]


def process(
    conf: Dict[str, Any],
    start_random_state: int,
    n_random_state: int,
):
    log_dir = os.getcwd()
    conf["log_dir"] = log_dir
    experiment = conf["experiment"]
    dataset = conf["dataset"]
    reward_function_type = conf["reward_function_type"]
    betas = conf["betas"]
    report_results_for_all_beta = conf["report_results_for_all_beta"]

    p = Pool(1)
    returns = []
    for random_state in range(start_random_state, n_random_state):
        return_ = p.apply_async(
            wrapper_evaluate_estimators, args=((conf, random_state),)
        )
        returns.append(return_)
    p.close()

    # aggregate result
    baseline_estimators_name = ["DM", "DR", "PIDR", "MDR", "on_policy"]
    baseline_performance = defaultdict(list)
    lips_performance = defaultdict(list)
    target_values, random_states = [], []

    for return_ in returns:
        baseline_dict, lips_dict, target_value, random_state_ = return_.get()

        for baseline_dict_, lips_dict_, target_value_ in zip(
            baseline_dict, lips_dict, target_value
        ):
            for estimator_name in baseline_estimators_name:
                estimation_ = baseline_dict_[estimator_name]
                baseline_performance[estimator_name].append(estimation_)

            if report_results_for_all_beta:
                for beta in betas:
                    estimation_ = lips_dict_[beta]
                    lips_performance[beta].append(estimation_)

            estimation_ = lips_dict_["best"]
            lips_performance["LDR (Ours)"].append(estimation_[1])
            lips_performance["best beta"].append(estimation_[0])

            target_values.append(target_value_)
            random_states.append(random_state_)

    df = pd.DataFrame()
    df["DM"] = baseline_performance["DM"]
    df["DR"] = baseline_performance["DR"]
    df["PIDR"] = baseline_performance["PIDR"]
    df["MDR"] = baseline_performance["MDR"]
    df["LDR (Ours)"] = lips_performance["LDR (Ours)"]
    df["best beta"] = lips_performance["best beta"]

    if report_results_for_all_beta:
        for beta in betas:
            df[f"LDR ({beta})"] = lips_performance[beta]

    df["on_policy"] = baseline_performance["on_policy"]
    df["target_value"] = target_values
    df["random_state"] = random_states

    path_ = Path(log_dir + f"/logs/{dataset}/{reward_function_type}/dr_variants")
    path_.mkdir(exist_ok=True, parents=True)

    if start_random_state > 0:
        prev_df = pd.read_csv(
            path_ / f"estimation_{experiment}_{start_random_state}.csv"
        )
        df = pd.concat([prev_df, df], axis=0)

    df = df.sort_values(["target_value", "random_state"])
    df.to_csv(path_ / f"estimation_{experiment}.csv", index=False)
    df.to_csv(path_ / f"estimation_{experiment}_{n_random_state}.csv", index=False)

    aggdf = (
        df[["DM", "DR", "PIDR", "MDR", "LDR (Ours)", "on_policy", "target_value"]]
        .groupby("target_value")
        .mean()
    )
    print(aggdf)


def wrapper_evaluate_estimators(args):
    conf, random_state = args
    conf["random_state"] = random_state
    target_param = conf["target_param"]

    baseline_dict, lips_dict, target_value = [], [], []
    if target_param in ["n_rounds", "slate_size"]:
        for value in conf[target_param]:
            value_ = int(value)

            conf_ = deepcopy(conf)
            conf_[target_param] = value_
            conf_["target_value"] = value_

            baseline_dict_, lips_dict_ = evaluate_estimators(**conf_)
            baseline_dict.append(baseline_dict_)
            lips_dict.append(lips_dict_)
            target_value.append(value_)
    else:
        conf["target_value"] = None

        baseline_dict_, lips_dict_ = evaluate_estimators(**conf)
        baseline_dict.append(baseline_dict_)
        lips_dict.append(lips_dict_)
        target_value.append(None)

    return baseline_dict, lips_dict, target_value, random_state


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    assert_configuration(cfg)
    conf = {
        "experiment": cfg.setting.experiment,
        "dataset": cfg.setting.dataset,
        "n_rounds": cfg.setting.n_rounds,  #
        "n_unique_action": cfg.setting.n_unique_action,
        "slate_size": cfg.setting.slate_size,
        "behavior_tau": cfg.setting.behavior_tau,
        "behavior_epsilon": cfg.setting.behavior_epsilon,
        "evaluation_epsilon": cfg.setting.evaluation_epsilon,
        "reward_type": cfg.setting.reward_type,
        "reward_function_type": cfg.setting.reward_function_type,
        "reward_std": cfg.setting.reward_std,
        "target_param": cfg.setting.target_param,
        "batch_size": cfg.model_params.batch_size,
        "learning_rate": cfg.model_params.learning_rate,
        "abstraction_type": cfg.model_params.abstraction_type,
        "abstraction_dimension": cfg.model_params.abstraction_dimension,  #
        "betas": np.array(list(cfg.model_params.beta)),
        "report_results_for_all_beta": cfg.model_params.report_results_for_all_beta,
        "device": cfg.model_params.device,
        "base_random_state": cfg.setting.base_random_state,
        "use_base_random_state": cfg.setting.use_base_random_state,
    }
    start_random_state = cfg.setting.start_random_state
    n_random_state = cfg.setting.n_random_state
    # script
    process(
        conf=conf,
        start_random_state=start_random_state,
        n_random_state=n_random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
