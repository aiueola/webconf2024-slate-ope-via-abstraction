"""Off-Policy Estimators for Slate Policies."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate sample-wise rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM) estimator.

    Parameters
    ----------
    estimator_name: str, default='dm'.
        Name of the estimator.

    """

    estimator_name: str = "dm"

    def _estimate_round_rewards(
        self,
        value_prediction: np.ndarray,
    ):
        """Estimate sample-wise rewards.

        Parameters
        ----------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        return value_prediction

    def estimate_policy_value(
        self,
        value_prediction: np.ndarray,
        **kwargs,
    ):
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            value_prediction=value_prediction,
        ).mean()

    def estimate_interval(
        self,
        value_prediction: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            value_prediction=value_prediction,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class InversePropensityScoring(BaseOffPolicyEstimator):
    """Inverse Propensity Scoring (IPS) estimator.

    Parameters
    ----------
    estimator_name: str, default='ips'.
        Name of the estimator.

    """

    estimator_name: str = "ips"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        iw = (evaluation_policy_prob_for_chosen_action / pscore).prod(axis=1)
        return iw * reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) estimator.

    Parameters
    ----------
    estimator_name: str, default='dr'.
        Name of the estimator.

    """

    estimator_name: str = "dr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        iw = (evaluation_policy_prob_for_chosen_action / pscore).prod(axis=1)
        return iw * (reward - value_prediction_for_chosen_action) + value_prediction

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class PseudoInverse(BaseOffPolicyEstimator):
    """PseudoInverse (PI) estimator.

    Parameters
    ----------
    estimator_name: str, default='pi'.
        Name of the estimator.

    """

    estimator_name: str = "pi"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        slate_size = pscore.shape[1]
        iw = (evaluation_policy_prob_for_chosen_action / pscore).sum(axis=1)
        return (iw - slate_size + 1) * reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class PseudoInverseDoublyRobust(BaseOffPolicyEstimator):
    """DR-style PseudoInverse (PI) estimator.

    Parameters
    ----------
    estimator_name: str, default='pi_dr'.
        Name of the estimator.

    """

    estimator_name: str = "pi_dr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        slate_size = pscore.shape[1]
        iw = (evaluation_policy_prob_for_chosen_action / pscore).sum(axis=1)
        return (iw - slate_size + 1) * (
            reward - value_prediction_for_chosen_action
        ) + value_prediction

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class LatentInversePropensityScoring(BaseOffPolicyEstimator):
    """Latent Inverse Propensity Scoring (LIPS) estimator.

    Parameters
    ----------
    estimator_name: str, default='lips'.
        Name of the estimator.

    """

    estimator_name: str = "lips"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        abstraction_iw: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        abstraction_iw: array-like, shape (n_rounds, )
            Abstraction importance weight.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return abstraction_iw * reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        abstraction_iw: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        abstraction_iw: array-like, shape (n_rounds, )
            Abstraction importance weight.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            abstraction_iw=abstraction_iw,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        abstraction_iw: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        abstraction_iw: array-like, shape (n_rounds, )
            Abstraction importance weight.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            abstraction_iw=abstraction_iw,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class LatentDoublyRobust(BaseOffPolicyEstimator):
    """OffCEM estimator with latent importance weight.

    Parameters
    ----------
    estimator_name: str, default='ldr'.
        Name of the estimator.

    """

    estimator_name: str = "ldr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        abstraction_iw: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        abstraction_iw: array-like, shape (n_rounds, )
            Abstraction importance weight.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return (
            abstraction_iw * (reward - value_prediction_for_chosen_action)
            + value_prediction
        )

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        abstraction_iw: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        abstraction_iw: array-like, shape (n_rounds, )
            Abstraction importance weight.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            abstraction_iw=abstraction_iw,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        abstraction_iw: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        abstraction_iw: array-like, shape (n_rounds, )
            Abstraction importance weight.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            abstraction_iw=abstraction_iw,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MarginalizedIPS(BaseOffPolicyEstimator):
    """Marginalized IPS estimator with sufficient abstraction.

    Parameters
    ----------
    estimator_name: str, default='mips'.
        Name of the estimator.

    """

    estimator_name: str = "mips"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        slate_size = pscore.shape[1]
        iw = (evaluation_policy_prob_for_chosen_action / pscore)[
            :, : slate_size // 2
        ].prod(axis=1)
        return iw * reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MarginalizedDR(BaseOffPolicyEstimator):
    """Marginalized DR estimator with sufficient abstraction.

    Parameters
    ----------
    estimator_name: str, default='mdr'.
        Name of the estimator.

    """

    estimator_name: str = "mdr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        slate_size = pscore.shape[1]
        iw = (evaluation_policy_prob_for_chosen_action / pscore)[
            :, : slate_size // 2
        ].prod(axis=1)
        return iw * (reward - value_prediction_for_chosen_action) + value_prediction

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        value_prediction_for_chosen_action: np.ndarray,
        value_prediction: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        value_prediction_for_chosen_action: array-like, shape (n_rounds, )
            Predicted value given context and the chosen action.

        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
            value_prediction_for_chosen_action=value_prediction_for_chosen_action,
            value_prediction=value_prediction,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class NaiveAverage(BaseOffPolicyEstimator):
    """Naive average estimator.

    Parameters
    ----------
    estimator_name: str, default='na'.
        Name of the estimator.

    """

    estimator_name: str = "na"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
    ) -> np.ndarray:
        """Estimate sample-wise rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_prob_for_chosen_action: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Slate-level reward, i.e., :math:`r_{i}`.

        pscore: array-like, shape (n_rounds, slate_size)
            Behavior policy pscore for the chosen action.

        evaluation_policy_prob_for_chosen_action: array-like, shape (n_rounds, slate_size)
            Evaluation policy action choice probability for the chosen action.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            pscore=pscore,
            evaluation_policy_prob_for_chosen_action=evaluation_policy_prob_for_chosen_action,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
