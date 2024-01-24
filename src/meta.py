from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.utils import check_random_state

from src.estimators import BaseOffPolicyEstimator
from src.learners import LatentRepresentationLearning, RewardLearning


BanditFeedback = Dict[str, Any]


@dataclass
class OffPolicyEvaluation:
    """Class to conduct slate OPE with multiple estimators simultaneously.

    Parameters
    -----------
    bandit_feedback: BanditFeedback
        Logged bandit data used in OPE of slate/ranking policies.

    ope_estimators: List[BaseSlateOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `obp.ope.BaseSlateOffPolicyEstimator`.

    abstraction_learner: LatentRepresentationLearning, default=None
        Slate abstraction leaner.

    reward_learner: RewardLearning, default=None
        Reward learner.

    batch_size: int, default=32
        Batch size.

    lr: float, default=1e-4
        Learning rate.

    """

    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseOffPolicyEstimator]
    abstraction_learner: Optional[LatentRepresentationLearning] = None
    reward_learner: Optional[RewardLearning] = None
    batch_size: int = (32,)
    learning_rate: float = 1e-4
    random_state: Optional[int] = None

    def __post_init__(
        self,
    ):
        self.estimators = {}
        for estimator_ in self.ope_estimators:
            self.estimators[estimator_.estimator_name] = estimator_

        self.random_ = check_random_state(self.random_state)

    def fit_abstraction_model(
        self,
        beta: float = 1.0,
        require_init: bool = False,
    ):
        self.abstraction_learner.fit(
            context=self.bandit_feedback["context"],
            action=self.bandit_feedback["action"],
            reward=self.bandit_feedback["reward"],
            n_epoch=1000 if require_init else 500,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            beta=beta,
            requre_init=require_init,
        )

    def fit_reward_model(
        self,
    ):
        self.reward_learner.cross_fitting_fit(
            context=self.bandit_feedback["context"],
            action=self.bandit_feedback["action"],
            reward=self.bandit_feedback["reward"],
            batch_size=self.batch_size,
            lr=self.learning_rate,
        )

    def _create_ope_inputs(
        self,
        evaluation_policy_slot_prob_all: Optional[List[np.ndarray]] = None,
        evaluation_policy_prob_for_chosen_action: Optional[np.ndarray] = None,
        label: Optional[np.ndarray] = None,
        obtain_reward_prediction: bool = False,
        obtain_abstraction_iw: bool = False,
    ):
        """Create input dictionary of OPE estimators.

        Parameters
        -----------
        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        label: array-like, shape (n_rounds, slate_size)
            Positive or negative labels of extreme classification.

        obtain_reward_prediction: bool, default=False
            Whether to obtain predicted reward.

        obtain_abstraction_iw: bool, default=False.
            Whether to obtain abstraction importance weight.

        Returns
        ----------
        input_dict: dict
            Dictionary containing inputs for OPE estimators.

        """
        input_dict = {
            "reward": self.bandit_feedback["reward"],
            "pscore": self.bandit_feedback["pscore"],
            "pscore_all": self.bandit_feedback["pscore_all"],
            "evaluation_policy_prob_for_chosen_action": evaluation_policy_prob_for_chosen_action,
            "evaluation_policy_slot_prob_all": evaluation_policy_slot_prob_all,
            "value_prediction": None,
            "abstraction_iw": None,
        }

        if obtain_reward_prediction:
            if evaluation_policy_slot_prob_all is None:
                raise ValueError(
                    "evaluation_policy_slot_prob_all must be given when obtain_reward_prediction is True"
                )
            (
                input_dict["value_prediction"],
                input_dict["value_prediction_for_chosen_action"],
            ) = self.reward_learner.cross_fitting_predict(
                context=self.bandit_feedback["context"],
                action=self.bandit_feedback["action"],
                evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
            )

        if obtain_abstraction_iw:
            if evaluation_policy_slot_prob_all is None:
                raise ValueError(
                    "evaluation_policy_slot_prob_all must be given when obtain_abstraction_iw is True"
                )
            if label is None:
                input_dict["abstraction_iw"] = self.abstraction_learner.predict(
                    context=self.bandit_feedback["context"],
                    action=self.bandit_feedback["action"],
                    pscore_all=self.bandit_feedback["pscore_all"],
                    evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
                )
            else:
                input_dict["abstraction_iw"] = self._calc_true_abstraction_iw(
                    label=label,
                    action=self.bandit_feedback["action"],
                    pscore_all=self.bandit_feedback["pscore_all"],
                    evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
                    candidate_actions=self.bandit_feedback["candidate_actions"],
                )

        return input_dict

    def _slope_hyperparam_tuning(
        self,
        betas: np.ndarray,
        policy_value_dict: Dict[float, float],
        ci_dict: Dict[float, Dict[str, float]],
        alpha: float = 0.05,
    ):
        """SLOPE hyperpameter tuning.

        Parameters
        -----------
        betas: array-like, shape (n_candidate_beta, )
            Set of hyperparameters that control the granularity of slate abstraction.

        policy_dict_dict: dict
            Policy value estimated by LIPS with each value of beta.

        ci_dict: dict
            Confidence interval for each beta.

        alpha: float, default=0.05
            Significance level.

        """
        cis, estimates = [], []
        for i, beta in enumerate(betas):
            ci = (
                ci_dict[beta][f"{100 * (1. - alpha)}% CI (upper)"]
                - ci_dict[beta][f"{100 * (1. - alpha)}% CI (lower)"]
            )
            mean = policy_value_dict[beta]

            if i == 0:
                cis.append(ci)
                estimates.append(mean)
                best_beta = betas[-1]
                continue

            if (
                np.abs(mean - np.array(estimates))
                > ci + (np.sqrt(6) - 1) * np.array(cis)
            ).any():
                best_beta = betas[i - 1]
                break

            else:
                cis.append(ci)
                estimates.append(mean)

        best_estimate = policy_value_dict[best_beta]
        policy_value_dict["best"] = (best_beta, best_estimate)
        return best_beta

    def estimate_policy_value_with_baseline_estimators(
        self,
        estimator_names: List[str],
        evaluation_policy_slot_prob_all: Optional[List[np.ndarray]] = None,
        evaluation_policy_prob_for_chosen_action: Optional[np.ndarray] = None,
        require_reward_prediction: bool = True,
    ):
        """Estimate policy value using baseline estimators.

        Parameters
        -----------
        estimator_names: list of str
            Set of estimator names.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        require_reward_prediction: bool, default=True
            Whether to predict reward for DM.

        Returns
        ----------
        policy_dict: dict
            Policy value estimated by baseline estimators.

        """
        if require_reward_prediction:
            self.fit_reward_model()

        estimator_inputs = self._create_ope_inputs(
            evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            obtain_reward_prediction=require_reward_prediction,
        )

        policy_value_dict = dict()
        for estimator_name in estimator_names:
            policy_value_dict[estimator_name] = self.estimators[
                estimator_name
            ].estimate_policy_value(**estimator_inputs)

        return policy_value_dict

    def estimate_policy_value_with_lips(
        self,
        estimator_name: str,
        evaluation_policy_slot_prob_all: List[np.ndarray],
        betas: np.ndarray,
        alpha: float = 0.05,
        report_results_for_all_beta: bool = False,
        require_reward_prediction: bool = False,
    ):
        """Estimate policy value using lips.

        estimator_name: str
            Estimator name.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        betas: array-like, shape (n_candidate_beta, )
            Set of hyperparameters that control the granularity of slate abstraction.

        alpha: float, default=0.05
            Significance level.

        report_results_for_all_beta: bool, default=False
            Report values for all betas.

        require_reward_prediction: bool, default=True
            Whether to predict reward for DM.

        Returns
        ----------
        policy_dict: dict
            Policy value estimated by lips.

        """
        betas.sort()
        policy_value_dict = dict()
        ci_dict = dict()

        for i, beta in enumerate(betas):
            self.fit_abstraction_model(beta, require_init=(i == 0))
            estimator_inputs = self._create_ope_inputs(
                evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
                obtain_abstraction_iw=True,
                obtain_reward_prediction=require_reward_prediction,
            )
            policy_value_dict[beta] = self.estimators[
                estimator_name
            ].estimate_policy_value(
                **estimator_inputs,
            )
            ci_dict[beta] = self.estimators[estimator_name].estimate_interval(
                **estimator_inputs,
                alpha=alpha,
            )

            best_beta = self._slope_hyperparam_tuning(
                betas=betas[: i + 1],
                policy_value_dict=policy_value_dict,
                ci_dict=ci_dict,
                alpha=alpha,
            )
            if (not report_results_for_all_beta) and (best_beta < beta):
                break

        return policy_value_dict
