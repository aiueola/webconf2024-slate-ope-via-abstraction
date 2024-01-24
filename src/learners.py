from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm
from typing import List, Optional
from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from src.models import (
    SlateRewardPredictor,
    DiscreteSlateAbstractionRewardPredictor,
    ContinuousSlateAbstractionRewardPredictor,
    DiscreteSlateEncoder,
    ContinuousSlateEncoder,
    DiscreteSlateDecoder,
    ContinuousSlateDecoder,
)
from src.utils import to_tensor


@dataclass
class LatentRepresentationLearning:
    """Latent Representation Learner.

    Parameters
    -----------
    dim_context: int, default=1
        Number of dimensions of context vectors.

    n_unique_actions: np.ndarray of shape (slate_size, )
        Number of unique actions at each slot.

    reward_type: str
        Reward type.

    abstraction_type: str, default="discrete"
        Type of slate abstraction. Either "discrete" or "continuous".

    n_latent_abstraction: int, default=10
        Number of discrete slate abstraction.

    dim_latent_abstraction: int, default=5
        Dimension of continuous slate abstraction.

    hidden_dim: int, default=100
        Dimension of the hidden layer.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    log_dir: str, default=None
        Logging directory.

    """

    dim_context: int
    n_unique_actions: np.ndarray
    reward_type: str
    abstraction_type: str = "discrete"
    n_latent_abstraction: int = 10
    dim_latent_abstraction: int = 5
    hidden_dim: int = 100
    device: str = "cuda"
    random_state: Optional[int] = None
    log_dir: Optional[str] = None

    def __post_init__(self):
        self._init_model()

    def _init_model(self):
        if self.device == "cuda:0":
            torch.cuda.manual_seed(self.random_state)

        torch.manual_seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

        if self.abstraction_type == "discrete":
            self.encoder = DiscreteSlateEncoder(
                dim_context=self.dim_context,
                n_unique_actions=self.n_unique_actions,
                n_latent_abstraction=self.n_latent_abstraction,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
            self.decoder = DiscreteSlateDecoder(
                dim_context=self.dim_context,
                n_unique_actions=self.n_unique_actions,
                n_latent_abstraction=self.n_latent_abstraction,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
            self.reward_predictor = DiscreteSlateAbstractionRewardPredictor(
                dim_context=self.dim_context,
                n_latent_abstraction=self.n_latent_abstraction,
                reward_type=self.reward_type,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

        else:
            self.encoder = ContinuousSlateEncoder(
                dim_context=self.dim_context,
                n_unique_actions=self.n_unique_actions,
                dim_latent_abstraction=self.dim_latent_abstraction,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
            self.decoder = ContinuousSlateDecoder(
                dim_context=self.dim_context,
                n_unique_actions=self.n_unique_actions,
                dim_latent_abstraction=self.dim_latent_abstraction,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
            self.reward_predictor = ContinuousSlateAbstractionRewardPredictor(
                dim_context=self.dim_context,
                dim_latent_abstraction=self.dim_latent_abstraction,
                reward_type=self.reward_type,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

    def _save_learning_curve(
        self,
        reward_prediction_loss: np.ndarray,
        reconstruction_loss: np.ndarray,
        kl_loss: np.ndarray,
        beta: float,
    ):
        n_epochs = len(reward_prediction_loss)
        x = np.arange(n_epochs)

        plt.style.use("ggplot")

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(3.75, 2.5 * 3),
        )

        axes[0].plot(x, reward_prediction_loss)
        axes[1].plot(x, reconstruction_loss)
        axes[2].plot(x, kl_loss)

        axes[0].set_ylabel("reward prediction loss")
        axes[1].set_ylabel("reconstruction loss")
        axes[2].set_ylabel("kl loss")

        for i in range(3):
            axes[i].set_xlabel("epochs")

        fig.tight_layout()

        path_ = Path(self.log_dir + "/learning_curve/")
        path_.mkdir(exist_ok=True, parents=True)

        if self.abstraction_type == "discrete":
            fig.savefig(
                path_
                / f"{self.abstraction_type}_{self.n_latent_abstraction}_{beta}.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                path_
                / f"{self.abstraction_type}_{self.dim_latent_abstraction}_{beta}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        n_epoch: int = 500,
        n_step_per_epoch: int = 100,
        test_ratio: float = 0.2,
        batch_size: int = 32,
        lr: float = 1e-4,
        beta: float = 1.0,
        reward_loss_weight: float = 1000.0,
        requre_init: bool = False,
    ):
        """Fit slate abstraction model.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        n_epoch: int, default=1000
            Number of epochs.

        n_step_per_epoch: int, default=100
            Number of gradient steps in an epoch.

        test_ratio: float, default=0.2
            Proportion of the test data.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        beta: float, default=1.0
            Hyperparameter that controls the granularity of abstraction and balances the bias-variance tradeoff.

        require_init: bool, default=False
            Whether to re-initialize the model params.

        """
        if requre_init:
            self._init_model()

        n_rounds, slate_size = action.shape
        test_size = int(n_rounds * test_ratio)
        train_size = n_rounds - test_size

        test_ids = self.random_.choice(n_rounds, size=test_size, replace=False)
        train_ids = np.setdiff1d(np.arange(n_rounds), test_ids)

        test_context = to_tensor(context[test_ids], device=self.device)
        test_action = to_tensor(action[test_ids], dtype=int, device=self.device)
        test_action_onehot = F.one_hot(
            test_action, num_classes=self.n_unique_actions.max()
        )
        test_reward = to_tensor(reward[test_ids], device=self.device)

        context = to_tensor(context[train_ids], device=self.device)
        action = to_tensor(action[train_ids], dtype=int, device=self.device)
        action_onehot = F.one_hot(action, num_classes=self.n_unique_actions.max())
        reward = to_tensor(reward[train_ids], device=self.device)

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        reward_predictor_optimizer = optim.Adam(
            self.reward_predictor.parameters(), lr=lr
        )

        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        reward_prediction_loss = np.zeros(n_epoch)
        reconstruction_loss = np.zeros(n_epoch)
        kl_loss = np.zeros(n_epoch)

        prev_test_loss_ = np.infty
        early_stopping_flg = np.zeros((5,))
        for epoch in range(n_epoch):
            for grad_step in range(n_step_per_epoch):
                idx_ = torch.randint(train_size, size=(batch_size,))
                context_ = context[idx_]
                action_ = action_onehot[idx_]
                reward_ = reward[idx_]

                latent_, latent_log_prob_ = self.encoder(context_, action_)
                reconstruction_log_prob_ = self.decoder(context_, latent_)
                reward_prediction_ = self.reward_predictor(context_, latent_)

                # reward prediction loss (bias)
                if self.reward_type == "binary":
                    reward_prediction_loss_ = (
                        bce_loss(reward_prediction_, reward_) * reward_loss_weight
                    )
                else:
                    reward_prediction_loss_ = (
                        mse_loss(reward_prediction_, reward_) * reward_loss_weight
                    )

                # slate reconstruction loss (bias)
                reconstruction_loss_ = torch.zeros(1, device=self.device)
                for l in range(slate_size):
                    reconstruction_loss_ += (
                        (reconstruction_log_prob_[l] * action_[:, l]).sum(axis=1).mean()
                    )

                # empirical KL loss with prior (variance)
                if self.abstraction_type == "discrete":
                    for l in range(slate_size):
                        kl_loss_ = (
                            latent_log_prob_ + math.log(self.n_latent_abstraction)
                        ).mean()
                else:
                    for l in range(slate_size):
                        kl_loss_ = (
                            latent_log_prob_
                            - (
                                -math.log((2 * math.pi) ** (1 / 2))
                                - torch.sqrt(latent_) / 2
                            ).sum(axis=1)
                        ).mean()

                loss_ = reward_prediction_loss_ - (
                    reconstruction_loss_ - beta * kl_loss_
                )

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                reward_predictor_optimizer.zero_grad()

                loss_.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()
                reward_predictor_optimizer.step()

                reward_prediction_loss[epoch] = reward_prediction_loss_.item()
                reconstruction_loss[epoch] = reconstruction_loss_.item()
                kl_loss[epoch] = kl_loss_.item()

            if epoch % 10 == 0:
                print(
                    f"epoch={epoch: >4}, "
                    f"reward_prediction_loss={reward_prediction_loss_.item():.3f}, "
                    f"reconstruction_loss={reconstruction_loss_.item():.3f}, "
                    f"kl_loss={kl_loss_.item():.3f}"
                )

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        pscore_all: List[np.ndarray],
        evaluation_policy_slot_prob_all: List[np.ndarray],
        n_samples_to_approximate: int = 1000,
    ):
        """Predict abstraction importance weight of the given context-action pair.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        pscore_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Behavior policy action choice probability for all actions at each slot.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        n_samples_to_approximate: int, default=1000
            Number of slates to draw to approximate the slate sampling process.

        Returns
        ----------
        abstraction_iw: np.ndarray of shape (n_rounds, )
            Importance weight for the sampled slate abstraction.

        """
        n_rounds = len(context)
        slate_size = len(pscore_all)

        context_ = to_tensor(context, device=self.device)
        action_ = to_tensor(action, dtype=int, device=self.device)
        action_onehot_ = F.one_hot(action_, num_classes=self.n_unique_actions.max())

        with torch.no_grad():
            latent_ = self.encoder(context_, action_onehot_)[0]

        behavior_abstraction_prob = np.zeros((n_rounds,))
        evaluation_abstraction_prob = np.zeros((n_rounds,))

        sampled_action_behavior = np.zeros(
            (n_rounds, n_samples_to_approximate, slate_size)
        )
        sampled_action_evaluation = np.zeros(
            (n_rounds, n_samples_to_approximate, slate_size)
        )
        for i in range(n_rounds):
            for l in np.arange(slate_size):
                sampled_action_behavior[i, :, l] = self.random_.choice(
                    self.n_unique_actions[l],
                    p=pscore_all[l][i],
                    size=n_samples_to_approximate,
                    replace=True,
                )
                sampled_action_evaluation[i, :, l] = self.random_.choice(
                    self.n_unique_actions[l],
                    p=evaluation_policy_slot_prob_all[l][i],
                    size=n_samples_to_approximate,
                    replace=True,
                )

        behavior_abstraction_prob = np.zeros((n_rounds,))
        evaluation_abstraction_prob = np.zeros((n_rounds,))
        batch_size = 1000000 // n_samples_to_approximate

        for i in tqdm(
            np.arange(n_rounds // batch_size),
            desc="[calc_abstraction_iw]",
            total=n_rounds // batch_size,
        ):
            context_enum_ = torch.repeat_interleave(
                context_[i * batch_size : (i + 1) * batch_size],
                n_samples_to_approximate,
                axis=0,
            )
            latent_enum_ = torch.repeat_interleave(
                latent_[i * batch_size : (i + 1) * batch_size],
                n_samples_to_approximate,
                axis=0,
            )
            sampled_action_behavior_ = to_tensor(
                sampled_action_behavior[i * batch_size : (i + 1) * batch_size].reshape(
                    (-1, slate_size)
                ),
                dtype=int,
                device=self.device,
            )
            sampled_action_behavior_onehot_ = F.one_hot(
                sampled_action_behavior_, num_classes=self.n_unique_actions.max()
            )
            sampled_action_evaluation_ = to_tensor(
                sampled_action_evaluation[
                    i * batch_size : (i + 1) * batch_size
                ].reshape((-1, slate_size)),
                dtype=int,
                device=self.device,
            )
            sampled_action_evaluation_onehot_ = F.one_hot(
                sampled_action_evaluation_, num_classes=self.n_unique_actions.max()
            )
            with torch.no_grad():
                behavior_abstraction_prob[i * batch_size : (i + 1) * batch_size] = (
                    self.encoder.calc_latent_prob(
                        context_enum_, sampled_action_behavior_onehot_, latent_enum_
                    )
                    .reshape((-1, n_samples_to_approximate))
                    .mean(axis=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                evaluation_abstraction_prob[i * batch_size : (i + 1) * batch_size] = (
                    self.encoder.calc_latent_prob(
                        context_enum_, sampled_action_evaluation_onehot_, latent_enum_
                    )
                    .reshape((-1, n_samples_to_approximate))
                    .mean(axis=1)
                    .detach()
                    .cpu()
                    .numpy()
                )

        abstraction_iw = evaluation_abstraction_prob / (
            behavior_abstraction_prob + 1e-10
        )
        return abstraction_iw

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_all: List[np.ndarray],
        evaluation_policy_slot_prob_all: List[np.ndarray],
        batch_size: int = 32,
        lr: float = 1e-4,
        beta: float = 1.0,
    ):
        """Fit slate abstraction model and predict abstraction importance weight of the given context-action pair.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        pscore_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Behavior policy action choice probability for all actions at each slot.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        Returns
        ----------
        abstraction_iw: np.ndarray of shape (n_rounds, )
            Importance weight for the sampled slate abstraction.

        """
        self.fit(
            context=context,
            action=action,
            reward=reward,
            beta=beta,
            batch_size=batch_size,
            lr=lr,
        )
        return self.predict(
            context=context,
            action=action,
            pscore_all=pscore_all,
            evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
        )

    def cross_fitting(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_all: List[np.ndarray],
        evaluation_policy_slot_prob_all: List[np.ndarray],
        batch_size: int = 32,
        lr: float = 1e-4,
        beta: float = 1.0,
        k_fold: int = 3,
    ):
        """Fit slate abstraction model and predict abstraction importance weight of the given context-action pair.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        pscore_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Behavior policy action choice probability for all actions at each slot.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        k_fold: int = 3
            Number of folds for cross-fitting.

        Returns
        ----------
        abstraction_iw: np.ndarray of shape (n_rounds, )
            Importance weight for the sampled slate abstraction.

        """
        n_rounds = len(context)
        kth_ids = np.array([(n_rounds + k) // k_fold for k in range(k_fold)])
        kth_ids = np.insert(kth_ids, 0, 0)
        kth_ids = np.cumsum(kth_ids)

        abstraction_iw = np.zeros(n_rounds)
        for k in range(k_fold):
            kth_complement_ids = np.setdiff1d(
                np.arange(n_rounds), np.arange(kth_ids[k], kth_ids[k + 1])
            )
            self.fit(
                context=context[kth_complement_ids],
                action=action[kth_complement_ids],
                reward=reward[kth_complement_ids],
                beta=beta,
                batch_size=batch_size,
                lr=lr,
            )
            abstraction_iw[kth_ids[k] : kth_ids[k + 1]] = self.predict(
                context=context[kth_ids[k] : kth_ids[k + 1]],
                action=action[kth_ids[k] : kth_ids[k + 1]],
                pscore_all=pscore_all[kth_ids[k] : kth_ids[k + 1]],
                evaluation_policy_slot_prob_all=[
                    evaluation_policy_slot_prob_all[l][kth_ids[k] : kth_ids[k + 1]]
                    for l in range(len(evaluation_policy_slot_prob_all))
                ],
            )

        return abstraction_iw


@dataclass
class RewardLearning:
    """Reward Learner.

    Parameters
    -----------
    dim_context: int, default=1
        Number of dimensions of context vectors.

    n_unique_actions: np.ndarray of shape (slate_size, )
        Number of unique actions at each slot.

    reward_type: str
        Reward type.

    hidden_dim: int, default=100
        Dimension of the hidden layer.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    log_dir: str, default=None
        Logging directory.

    """

    dim_context: int
    n_unique_actions: np.ndarray
    reward_type: str
    hidden_dim: int = 100
    device: str = "cuda"
    random_state: Optional[int] = None
    log_dir: Optional[str] = None

    def __post_init__(self):
        self._init_model()

    def _init_model(self):
        if self.device == "cuda:0":
            torch.cuda.manual_seed(self.random_state)

        torch.manual_seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

        self.reward_predictor = SlateRewardPredictor(
            dim_context=self.dim_context,
            n_unique_actions=self.n_unique_actions,
            reward_type=self.reward_type,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.reward_predictors = {}

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        model_id: int,
        n_epoch: int = 500,
        n_step_per_epoch: int = 100,
        test_ratio: float = 0.2,
        batch_size: int = 32,
        lr: float = 1e-4,
        reward_loss_weight: float = 100.0,
    ):
        """Fit reward prediction model.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        model_id: int
            Index to the model.

        n_epoch: int, default=1000
            Number of epochs.

        n_step_per_epoch: int, default=100
            Number of gradient steps in an epoch.

        test_ratio: float, default=0.2
            Proportion of the test data.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        """
        n_rounds, slate_size = action.shape
        test_size = int(n_rounds * test_ratio)
        train_size = n_rounds - test_size

        test_ids = self.random_.choice(n_rounds, size=test_size, replace=False)
        train_ids = np.setdiff1d(np.arange(n_rounds), test_ids)

        test_context = to_tensor(context[test_ids], device=self.device)
        test_action = to_tensor(action[test_ids], dtype=int, device=self.device)
        test_action = F.one_hot(test_action, num_classes=self.n_unique_actions.max())
        test_reward = to_tensor(reward[test_ids], device=self.device)

        context = to_tensor(context[train_ids], device=self.device)
        action = to_tensor(action[train_ids], dtype=int, device=self.device)
        action_onehot = F.one_hot(action, num_classes=self.n_unique_actions.max())
        reward = to_tensor(reward[train_ids], device=self.device)

        if model_id not in self.reward_predictors:
            self.reward_predictors[model_id] = deepcopy(self.reward_predictor)

        optimizer = optim.Adam(self.reward_predictors[model_id].parameters(), lr=lr)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        prev_test_loss_ = np.infty
        early_stopping_flg = np.zeros((5,))
        for epoch in range(n_epoch):
            for grad_step in range(n_step_per_epoch):
                idx_ = torch.randint(train_size, size=(batch_size,))
                context_ = context[idx_]
                action_ = action_onehot[idx_]
                reward_ = reward[idx_]

                reward_prediction_ = self.reward_predictors[model_id](context_, action_)

                if self.reward_type == "binary":
                    loss_ = bce_loss(reward_prediction_, reward_)
                else:
                    loss_ = mse_loss(reward_prediction_, reward_) * reward_loss_weight

                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()

            with torch.no_grad():
                test_reward_prediction = self.reward_predictors[model_id](
                    test_context, test_action
                )
                if self.reward_type == "binary":
                    test_loss_ = bce_loss(test_reward_prediction, test_reward)
                else:
                    test_loss_ = (
                        mse_loss(test_reward_prediction, test_reward)
                        * reward_loss_weight
                    )

            if prev_test_loss_ < test_loss_:
                early_stopping_flg[epoch % 5] = True
            else:
                early_stopping_flg[epoch % 5] = False

            if early_stopping_flg.all():
                print(f"early stopping at epoch={epoch: >4}")
                break

            if epoch % 10 == 0:
                print(
                    f"epoch={epoch: >4}, "
                    f"train_loss={loss_.item():.3f}, "
                    f"test_loss={test_loss_.item():.3f}"
                )

            prev_test_loss_ = test_loss_

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        evaluation_policy_slot_prob_all: List[np.ndarray],
        model_id: int,
        n_samples_to_approximate: int = 1000,
    ):
        """Predict mean reward function for all slates.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        model_id: int
            Index to the model.

        n_samples_to_approximate: int, default=10000
            Number of slates to draw to approximate the slate sampling process.

        Returns
        ----------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        value_prediction_for_chosen_action: array-like of shape (n_rounds, )
            Predicted value given context and the action chosen by the behavior policy.

        """
        n_rounds = len(context)
        slate_size = len(evaluation_policy_slot_prob_all)
        context_ = to_tensor(context, device=self.device)

        sampled_action = np.zeros((n_rounds, n_samples_to_approximate, slate_size))
        for i in range(n_rounds):
            for l in np.arange(slate_size):
                sampled_action[i, :, l] = self.random_.choice(
                    self.n_unique_actions[l],
                    p=evaluation_policy_slot_prob_all[l][i],
                    size=n_samples_to_approximate,
                    replace=True,
                )

        value_prediction = np.zeros((n_rounds,))
        batch_size = 1000000 // n_samples_to_approximate

        for i in tqdm(
            np.arange(n_rounds // batch_size),
            desc="[predict_value]",
            total=n_rounds // batch_size,
        ):
            context_enum_ = torch.repeat_interleave(
                context_[i * batch_size : (i + 1) * batch_size],
                n_samples_to_approximate,
                axis=0,
            )
            sampled_action_ = to_tensor(
                sampled_action[i * batch_size : (i + 1) * batch_size].reshape(
                    (-1, slate_size)
                ),
                dtype=int,
                device=self.device,
            )
            sampled_action_onehot_ = F.one_hot(
                sampled_action_, num_classes=self.n_unique_actions.max()
            )

            with torch.no_grad():
                value_prediction[i * batch_size : (i + 1) * batch_size] = (
                    self.reward_predictors[model_id](
                        context_enum_, sampled_action_onehot_
                    )
                    .reshape((-1, n_samples_to_approximate))
                    .mean(axis=1)
                    .detach()
                    .cpu()
                    .numpy()
                )

        action_ = to_tensor(action, dtype=int, device=self.device)
        action_onehot_ = F.one_hot(action_, num_classes=self.n_unique_actions.max())

        with torch.no_grad():
            value_prediction_for_chosen_action = (
                self.reward_predictors[model_id](context_, action_onehot_)
                .detach()
                .cpu()
                .numpy()
            )

        return value_prediction, value_prediction_for_chosen_action

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_slot_prob_all: List[np.ndarray],
        batch_size: int = 32,
        lr: float = 1e-4,
    ):
        """Fit reward prediction model and predict mean reward function for all slates.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        Returns
        ----------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        value_prediction_for_chosen_action: array-like of shape (n_rounds, )
            Predicted value given context and the action chosen by the behavior policy.

        """
        self.fit(
            context=context,
            action=action,
            reward=reward,
            batch_size=batch_size,
            model_id=0,
            lr=lr,
        )
        return self.predict(
            context=context,
            action=action,
            evaluation_policy_slot_prob_all=evaluation_policy_slot_prob_all,
            model_id=0,
        )

    def cross_fitting_fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        batch_size: int = 32,
        lr: float = 1e-4,
        k_fold: int = 3,
    ):
        """Fit slate abstraction model and predict abstraction importance weight of the given context-action pair.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        k_fold: int = 3
            Number of folds for cross-fitting.

        """
        n_rounds, slate_size = action.shape

        kth_ids = np.array([(n_rounds + k) // k_fold for k in range(k_fold)])
        kth_ids = np.insert(kth_ids, 0, 0)
        kth_ids = np.cumsum(kth_ids)

        for k in range(k_fold):
            kth_complement_ids = np.setdiff1d(
                np.arange(n_rounds), np.arange(kth_ids[k], kth_ids[k + 1])
            )
            self.fit(
                context=context[kth_complement_ids],
                action=action[kth_complement_ids],
                reward=reward[kth_complement_ids],
                model_id=k,
                batch_size=batch_size,
                lr=lr,
            )

    def cross_fitting_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        evaluation_policy_slot_prob_all: List[np.ndarray],
        k_fold: int = 3,
    ):
        """Fit slate abstraction model and predict abstraction importance weight of the given context-action pair.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        k_fold: int = 3
            Number of folds for cross-fitting.

        Returns
        ----------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        value_prediction_for_chosen_action: array-like of shape (n_rounds, )
            Predicted value given context and the action chosen by the behavior policy.

        """
        n_rounds, slate_size = action.shape

        kth_ids = np.array([(n_rounds + k) // k_fold for k in range(k_fold)])
        kth_ids = np.insert(kth_ids, 0, 0)
        kth_ids = np.cumsum(kth_ids)

        value_prediction = np.zeros(
            n_rounds,
        )
        value_prediction_for_chosen_action = np.zeros(
            n_rounds,
        )

        for k in range(k_fold):
            (
                value_prediction[kth_ids[k] : kth_ids[k + 1]],
                value_prediction_for_chosen_action[kth_ids[k] : kth_ids[k + 1]],
            ) = self.predict(
                context=context[kth_ids[k] : kth_ids[k + 1]],
                action=action[kth_ids[k] : kth_ids[k + 1]],
                evaluation_policy_slot_prob_all=[
                    evaluation_policy_slot_prob_all[l][kth_ids[k] : kth_ids[k + 1]]
                    for l in range(slate_size)
                ],
                model_id=k,
            )

        return value_prediction, value_prediction_for_chosen_action

    def cross_fitting(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_slot_prob_all: List[np.ndarray],
        batch_size: int = 32,
        lr: float = 1e-4,
        k_fold: int = 3,
    ):
        """Fit slate abstraction model and predict abstraction importance weight of the given context-action pair.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds, slate_size)
            Action chosen by the behavior policy.

        reward: array-like, shape (n_rounds, )
            Reward observed for the presented slate.

        evaluation_policy_slot_prob_all: list of array-like, shape (n_rounds, n_unique_action[l])
            Evaluation policy action choice probability for all actions at each slot.

        batch_size: int, default=32
            Batch size.

        lr: float, default=1e-4
            Learning rate.

        k_fold: int = 3
            Number of folds for cross-fitting.

        Returns
        ----------
        value_prediction: array-like of shape (n_rounds, )
            Predicted value given context.

        value_prediction_for_chosen_action: array-like of shape (n_rounds, )
            Predicted value given context and the action chosen by the behavior policy.

        """
        n_rounds, slate_size = action.shape

        kth_ids = np.array([(n_rounds + k) // k_fold for k in range(k_fold)])
        kth_ids = np.insert(kth_ids, 0, 0)
        kth_ids = np.cumsum(kth_ids)

        value_prediction = np.zeros(
            n_rounds,
        )
        value_prediction_for_chosen_action = np.zeros(
            n_rounds,
        )

        for k in range(k_fold):
            kth_complement_ids = np.setdiff1d(
                np.arange(n_rounds), np.arange(kth_ids[k], kth_ids[k + 1])
            )
            self.fit(
                context=context[kth_complement_ids],
                action=action[kth_complement_ids],
                reward=reward[kth_complement_ids],
                model_id=k,
                batch_size=batch_size,
                lr=lr,
            )
            (
                value_prediction[kth_ids[k] : kth_ids[k + 1]],
                value_prediction_for_chosen_action[kth_ids[k] : kth_ids[k + 1]],
            ) = self.predict(
                context=context[kth_ids[k] : kth_ids[k + 1]],
                action=action[kth_ids[k] : kth_ids[k + 1]],
                evaluation_policy_slot_prob_all=[
                    evaluation_policy_slot_prob_all[l][kth_ids[k] : kth_ids[k + 1]]
                    for l in range(slate_size)
                ],
                model_id=k,
            )

        return value_prediction, value_prediction_for_chosen_action
