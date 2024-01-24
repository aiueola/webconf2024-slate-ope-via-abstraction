"""Class for Generating Semi-synthetic Logged Bandit Data for Slate Policies."""
from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import List
from typing import Callable
from tqdm import tqdm
from pathlib import Path
import random

import numpy as np
from scipy.stats import truncnorm
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

import torch
from torch import optim

from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sigmoid, softmax
from obp.dataset.base import BaseBanditDataset

from src.utils import to_tensor, torch_seed

from .classifier import Classifier


@dataclass
class SemiSyntheticSlateBanditDataset(BaseBanditDataset):
    """Class for synthesizing slate bandit dataset.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we can resample logged bandit data from the same data generating distribution.
    This can be used to estimate confidence intervals of the performances of Slate OPE estimators.

    Parameters
    -----------
    slate_size: int (> 1)
        Length of a list of actions, slate size.

    n_unique_action: np.ndarray of shape (slate_size, )
        Number of unique actions at each slot.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary', rewards are sampled from the Bernoulli distribution.
        When 'continuous', rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function`.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function defining the expected reward function for each given slate-context pair,
        i.e., :math:`q: \\mathcal{X} \\times \\mathcal{S} \\rightarrow \\mathbb{R}`.

    reward_noise: float, default=1.0
        Noise level of reward.

    epsilon: float, default=None
        Exploration hyperparameter of the behavior policy.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic slate bandit dataset.

    dataset_name: str, default="delicious"
        Name of the base dataset.

    """

    slate_size: int
    n_unique_action: np.ndarray
    reward_type: str = "binary"
    reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ] = None
    reward_std: float = 0.1
    epsilon: float = 0.01
    tau: float = 1.0
    device: str = "cuda:0"
    base_random_state: int = 12345
    dataset_name: str = "delicious"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.slate_size, "slate_size", int, min_val=2)
        check_array(self.n_unique_action, name="n_unique_action", expected_dim=1)
        if self.n_unique_action.min() < 2:
            raise ValueError("n_unique_action must be equal to or more than 2")
        if len(self.n_unique_action) != self.slate_size:
            raise ValueError(
                "n_unique_actions.shape[0] must be equal to slate_size, but found False"
            )

        if self.n_unique_action.min() < 1:
            raise ValueError("dim_action_context must be equal to or more than 1")
        if len(self.n_unique_action) != self.slate_size:
            raise ValueError(
                "dim_action_context.shape[0] must be equal to slate_size, but found False"
            )
        self.random_ = check_random_state(self.base_random_state)
        if self.reward_type not in [
            "binary",
            "continuous",
        ]:
            raise ValueError(
                f"`reward_type` must be either 'binary' or 'continuous', but {self.reward_type} is given."
            )
        check_scalar(self.reward_std, "reward_std", float, min_val=0.0)
        check_scalar(self.tau, "tau", float)

        if self.reward_type == "continuous":
            self.reward_min = 0
            self.reward_max = 1e10

        self._load_dataset()
        self._train_base_classifier()

    def set_random_state(
        self,
        random_state: int,
    ):
        self.random_state = random_state
        self.random_ = check_random_state(random_state)

    def _load_text_dataset(
        self,
    ):
        path_processed = Path(f"raw_dataset/{self.dataset_name}/X.trn.npy")
        path_ = f"raw_dataset/{self.dataset_name}/"

        if path_processed.exists():
            X_train = np.load(path_ + "X.trn.npy", allow_pickle=True)
            X_test = np.load(path_ + "X.tst.npy", allow_pickle=True)
            Y_train = np.load(path_ + "Y.trn.npy", allow_pickle=True)
            Y_test = np.load(path_ + "Y.tst.npy", allow_pickle=True)

        else:
            all_raw_context = []
            with open(path_ + "X.trn.txt", "r") as f:
                for i, line in enumerate(f):
                    all_raw_context.append(line)
                n_train = i + 1
            with open(path_ + "X.tst.txt", "r") as f:
                for i, line in enumerate(f):
                    all_raw_context.append(line)

            all_labels = np.zeros((len(all_raw_context), 31000))
            with open(path_ + "Y.trn.txt", "rb") as f:
                for i, line in enumerate(f):
                    line = line.decode("utf-8").split(",")
                    line = np.array([int(word) for word in line])
                    all_labels[i, line] = True
            with open(path_ + "Y.tst.txt", "rb") as f:
                for i, line in enumerate(f):
                    line = line.decode("utf-8").split(",")
                    line = np.array([int(word) for word in line])
                    all_labels[i + n_train, line] = True

            n_positive_context = all_labels.sum(axis=0)
            # disable labels that are positive for more than 1000 actions (too easy labels)
            n_positive_context[np.where(n_positive_context > 1000)] = 0
            # extract top 1000 dense labels
            action_ids = np.argpartition(-n_positive_context, 1000)[:1000]

            all_labels = all_labels[:, action_ids]
            val_user_ids = all_labels.sum(axis=1) > 0
            n_val_train = val_user_ids[:n_train].sum()

            Y = all_labels[val_user_ids]
            val_user_ids = np.arange(len(all_labels))[val_user_ids]

            all_val_raw_context = []
            for i in val_user_ids:
                all_val_raw_context.append(all_raw_context[i])

            print("downloading bert encoder..")
            bert = SentenceTransformer("bert-base-nli-mean-tokens")
            bert.max_seq_length = 512

            embeddings = bert.encode(all_val_raw_context)
            pca = PCA(n_components=20)
            X = pca.fit_transform(embeddings)

            X_train, X_test = X[:n_val_train], X[n_val_train:]
            Y_train, Y_test = Y[:n_val_train], Y[n_val_train:]

            np.save(path_ + "X.trn.npy", X_train)
            np.save(path_ + "X.tst.npy", X_test)
            np.save(path_ + "Y.trn.npy", Y_train)
            np.save(path_ + "Y.tst.npy", Y_test)

        return X_train, X_test, Y_train, Y_test

    def _load_dataset(
        self,
    ):
        random.seed(self.base_random_state)
        np.random.seed(self.base_random_state)
        torch_seed(self.base_random_state, device=self.device)

        if self.dataset_name in ["bibtex", "delicious"]:
            (
                self.X_test,
                self.Y_test,
                _,
                _,
            ) = load_dataset(self.dataset_name, "train")

            (
                self.X_train,
                self.Y_train,
                _,
                _,
            ) = load_dataset(self.dataset_name, "test")

        elif self.dataset_name in ["eurlex", "wiki", "amazon"]:
            (
                self.X_train,
                self.X_test,
                self.Y_train,
                self.Y_test,
            ) = self._load_text_dataset()

        self.n_train_dataset = self.X_train.shape[0]
        self.n_test_dataset = self.X_test.shape[0]
        self.dim_context = self.X_train.shape[1]
        self.n_labels = self.Y_train.shape[1]

        self.candidate_actions = []
        for l in range(self.slate_size):
            self.candidate_actions.append(
                self.random_.choice(
                    self.n_labels,
                    size=self.n_unique_action[l],
                    replace=False,
                )
            )

    def _train_base_classifier(
        self,
        n_epoch: int = 300,
        n_step_per_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 1e-4,
    ):
        random.seed(self.base_random_state)
        np.random.seed(self.base_random_state)
        torch_seed(self.base_random_state, device=self.device)

        X_train, X_val, Y_train, Y_val = train_test_split(
            self.X_train, self.Y_train, train_size=0.8
        )

        if self.dataset_name in ["bibtex", "delicious"]:
            X_train = to_tensor(X_train.A, device=self.device)
            X_val = to_tensor(X_val.A, device=self.device)
        else:
            X_train = to_tensor(X_train, device=self.device)
            X_val = to_tensor(X_val, device=self.device)

        self.base_classifier = []
        for l in range(self.slate_size):
            classifier_ = Classifier(
                dim_context=self.dim_context,
                n_unique_action=self.n_unique_action[l],
            ).to(self.device)

            if self.dataset_name in ["bibtex", "delicious"]:
                Y_train_ = to_tensor(
                    Y_train[:, self.candidate_actions[l]].A, device=self.device
                )
                Y_val_ = to_tensor(
                    Y_val[:, self.candidate_actions[l]].A, device=self.device
                )
            else:
                Y_train_ = to_tensor(
                    Y_train[:, self.candidate_actions[l]], device=self.device
                )
                Y_val_ = to_tensor(
                    Y_val[:, self.candidate_actions[l]], device=self.device
                )

            test_reward_ = Y_val_ * 0.8 + 0.2

            # REINFORCE algorithm
            optimizer_ = optim.Adam(classifier_.parameters(), lr=lr)

            prev_test_loss_ = -np.infty
            early_stopping_flg = np.zeros((5,))
            for epoch in range(n_epoch):
                for grad_step in range(n_step_per_epoch):
                    idx_ = torch.randint(X_train.shape[0], size=(batch_size,))
                    context_ = X_train[idx_]
                    reward_ = Y_train_[idx_] * 0.8 + 0.2

                    action_, log_prob_ = classifier_(context_)
                    loss_ = -((reward_ * action_).sum(axis=1) * log_prob_).mean()

                    optimizer_.zero_grad()
                    loss_.backward()
                    optimizer_.step()

                with torch.no_grad():
                    test_action_, test_log_prob_ = classifier_(X_val)
                    test_loss_ = (
                        (test_reward_ * test_action_).sum(axis=1) * test_log_prob_
                    ).mean()

                if prev_test_loss_ > test_loss_:
                    early_stopping_flg[epoch % 5] = True
                else:
                    early_stopping_flg[epoch % 5] = False

                if early_stopping_flg.all():
                    print(f"early stopping at epoch={epoch: >4}")
                    break

                if epoch % 10 == 0:
                    print(
                        f"epoch={epoch: >4}, "
                        f"train_loss={-loss_.item():.3f}, "
                        f"test_loss={test_loss_:.3f}"
                    )

                prev_test_loss_ = test_loss_

            self.base_classifier.append(classifier_)

    def obtain_policy_logit(
        self,
        context: np.ndarray,
    ):
        """Obtain logit value.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        Returns
        ----------
        greedy_flg_: list of array-like, shape (n_rounds, n_unique_action[l])
            Greedy action (one-hot vector) predicted by the base classifier.

        """
        context_ = to_tensor(context, device=self.device)

        logits = []
        for l in range(self.slate_size):
            with torch.no_grad():
                logit_ = self.base_classifier[l].logit(context_).cpu().detach().numpy()
                logits.append(logit_)

        return logits

    def sample_action_and_obtain_pscore(
        self,
        policy_logit_: List[np.ndarray],
        epsilon: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sample action and obtain the three variants of the propensity scores.

        Parameters
        ----------
        policy_logit_: list of array-like, shape (n_rounds, n_unique_action[l])
            Logit values given context (:math:`x`).

        epsilon: float, default=None
            Exploration rate of the epsilon-greedy policy. When `None` is given, the softmax policy is used instead.

        Returns
        ----------
        action: array-like, shape (n_rounds, slate_size)
            Actions sampled by the behavior policy.
            Actions sampled within slate `i` is stored in `action[`i` * `slate_size`: (`i + 1`) * `slate_size`]`.

        pscore: array-like, shape (n_rounds, slate_size)
            Probabilities of choosing the slot action (:math:`a_l`) given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,l} | x_{i} )`.

        """
        if len(policy_logit_) != self.slate_size:
            raise ValueError(
                "len(policy_logit_) must be equal to slate_size, but found False"
            )
        for l in range(self.slate_size):
            check_array(
                array=policy_logit_[l],
                name="policy_logit_[l]",
                expected_dim=2,
            )
            if policy_logit_[l].shape[1] != self.n_unique_action[l]:
                raise ValueError(
                    "Expected policy_logit_[l].shape[1] == self.n_unique_action[l], but found False"
                )
        if epsilon is not None:
            check_scalar(epsilon, "epsilon", float, min_val=0.0)

        n_rounds = policy_logit_[-1].shape[0]
        action = np.zeros((n_rounds, self.slate_size), dtype=int)
        pscore = np.zeros((n_rounds, self.slate_size))

        pscore_all = []
        if epsilon is None:
            for l in range(self.slate_size):
                # prob = softmax(self.tau * policy_logit_[l])
                prob = (1 - self.epsilon) * softmax(
                    self.tau * policy_logit_[l]
                ) + self.epsilon / self.n_unique_action[l]
                pscore_all.append(prob)
        else:
            for l in range(self.slate_size):
                greedy_flg = (
                    np.tile(
                        (policy_logit_[l]).max(axis=1), (self.n_unique_action[l], 1)
                    ).T
                    == policy_logit_[l]
                )
                prob = (1 - epsilon) * greedy_flg + epsilon / self.n_unique_action[l]
                pscore_all.append(prob)

        for i in tqdm(
            np.arange(n_rounds),
            desc="[sample_action_and_obtain_pscore]",
            total=n_rounds,
        ):
            for l in np.arange(self.slate_size):
                action[i, l] = self.random_.choice(
                    self.n_unique_action[l], p=pscore_all[l][i], replace=False
                )
                pscore[i, l] = pscore_all[l][i, action[i, l]]

        return action, pscore

    def sample_reward_given_expected_reward(
        self, expected_reward_factual: np.ndarray
    ) -> np.ndarray:
        """Sample reward variables given actions observed at each slot.

        Parameters
        ------------
        expected_reward_factual: array-like, shape (n_rounds, )
            Expected rewards given observed actions and contexts.

        Returns
        ----------
        reward: array-like, shape (n_rounds, )
            Sampled rewards.

        """
        check_array(
            array=expected_reward_factual,
            name="expected_reward_factual",
            expected_dim=1,
        )

        n_rounds = expected_reward_factual.shape[0]
        reward = np.zeros(n_rounds)

        if self.reward_type == "binary":
            for i in range(n_rounds):
                reward[i] = self.random_.binomial(n=1, p=expected_reward_factual[i])
        else:  # "continuous"
            reward = np.zeros(expected_reward_factual.shape)
            for pos_ in np.arange(self.slate_size):
                mean = expected_reward_factual
                a = (self.reward_min - mean) / self.reward_std
                b = (self.reward_max - mean) / self.reward_std
                reward = truncnorm.rvs(
                    a=a,
                    b=b,
                    loc=mean,
                    scale=self.reward_std,
                    random_state=self.random_state,
                )

        return reward

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
    ) -> BanditFeedback:
        """Obtain batch logged bandit data.

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Synthesized slate logged bandit dataset.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        idx = self.random_.choice(self.n_test_dataset, size=n_rounds, replace=True)

        if self.dataset_name in ["bibtex", "delicious"]:
            context = self.X_test[idx].A
            label = self.Y_test[idx].A
        else:
            context = self.X_test[idx]
            label = self.Y_test[idx]

        # sample actions for each round based on the behavior policy
        policy_logit_ = self.obtain_policy_logit(context)
        # sample actions and calculate the three variants of the propensity scores
        (
            action,
            pscore,
        ) = self.sample_action_and_obtain_pscore(
            policy_logit_=policy_logit_,
        )
        # pscore for all slates
        pscore_all = self.calc_pscore_all(
            policy_logit_=policy_logit_,
        )
        # calc expected reward
        expected_reward_factual = self.reward_function(
            label=label,
            action=action,
            candidate_actions=self.candidate_actions,
            random_state=self.base_random_state,
        )
        # sample reward
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual
        )

        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_unique_action,
            context=context,
            label=label,
            action=action,
            reward=reward,
            expected_reward_factual=expected_reward_factual,
            pscore=pscore,
            pscore_all=pscore_all,
            candidate_actions=self.candidate_actions,
        )

    def calc_pscore_given_action(
        self,
        action: np.ndarray,
        policy_logit_: List[np.ndarray],
        epsilon: Optional[float] = None,
    ):
        """Calculate the action choice probability for the given action.

        Parameters
        ------------
        action: array-like, (n_rounds, slate_size)
            Action chosen by the behavior policy.

        policy_logit_: list of array-like, (n_rounds, n_unique_actions[l])
            Logit values to define the evaluation policy.

        epsilon: float, default=None
            Exploration rate of the epsilon-greedy policy.
            When `None` is given, the softmax policy is used instead.

        Returns
        ----------
        pscore: array-like, shape (n_rounds, slate_size)
            Probabilities of choosing the slot action (:math:`a_l`) given context (:math:`x`),
            i.e., :math:`\\pi(a_{i,l} | x_{i} )`.

        """
        check_array(array=action, name="action", expected_dim=2)
        if action.shape[1] != self.slate_size:
            raise ValueError(
                "action.shape[1] must be equal to slate_size, but found False"
            )
        if len(policy_logit_) != self.slate_size:
            raise ValueError(
                "len(policy_logit_) must be equal to slate_size, but found False"
            )
        for l in range(self.slate_size):
            check_array(
                array=policy_logit_[l],
                name="policy_logit_[l]",
                expected_dim=2,
            )
            if len(action) != len(policy_logit_[l]):
                raise ValueError(
                    "the lengths of `action` and `policy_logit_[l]` must be the same"
                )
            if policy_logit_[l].shape[1] != self.n_unique_action[l]:
                raise ValueError(
                    "Expected policy_logit_[l].shape[1] == self.n_unique_action[l], but found False"
                )
        if epsilon is not None:
            check_scalar(epsilon, "epsilon", float, min_val=0.0)

        n_rounds = action.shape[0]
        pscore = np.zeros((n_rounds, self.slate_size))

        if epsilon is None:
            for l in range(self.slate_size):
                # prob = softmax(self.tau * policy_logit_[l])
                prob = (1 - self.epsilon) * softmax(
                    self.tau * policy_logit_[l]
                ) + self.epsilon / self.n_unique_action[l]
                pscore[:, l] = prob[np.arange(n_rounds), action[:, l]]
        else:
            for l in range(self.slate_size):
                greedy_flg = policy_logit_[l].argmax(axis=1) == action[:, l]
                pscore[:, l] = (
                    1 - epsilon
                ) * greedy_flg + epsilon / self.n_unique_action[l]

        return pscore

    def calc_pscore_all(
        self,
        policy_logit_: List[np.ndarray],
        epsilon: Optional[float] = None,
    ):
        """Calculate the probability of choosing slate for all possible slates.

        Parameters
        -----------
        policy_logit_: list of array-like, (n_rounds, n_unique_actions[l])
            Logit values to define the evaluation policy.

        epsilon: float, default=None
            Exploration rate of the epsilon-greedy policy.
            When `None` is given, the softmax policy is used instead.

        Returns
        ----------
        pscore_all: list of np.ndarray of shape (n_rounds, n_unique_actions[l])
            Probability of choosing an action at each slot.

        """
        if len(policy_logit_) != self.slate_size:
            raise ValueError(
                "len(policy_logit_) must be equal to slate_size, but found False"
            )
        for l in range(self.slate_size):
            check_array(
                array=policy_logit_[l],
                name="policy_logit_[l]",
                expected_dim=2,
            )
            if policy_logit_[l].shape[1] != self.n_unique_action[l]:
                raise ValueError(
                    "Expected policy_logit_[l].shape[1] == self.n_unique_action[l], but found False"
                )
        if epsilon is not None:
            check_scalar(epsilon, "epsilon", float, min_val=0.0)

        pscore_all = []
        if epsilon is None:
            for l in tqdm(
                np.arange(self.slate_size),
                desc="[calc_pscore_all]",
                total=self.slate_size,
            ):
                # prob = softmax(self.tau * policy_logit_[l])
                prob = (1 - self.epsilon) * softmax(
                    self.tau * policy_logit_[l]
                ) + self.epsilon / self.n_unique_action[l]
                pscore_all.append(prob)
        else:
            for l in tqdm(
                np.arange(self.slate_size),
                desc="[calc_pscore_all]",
                total=self.slate_size,
            ):
                greedy_flg = (
                    np.tile(
                        (policy_logit_[l]).max(axis=1), (self.n_unique_action[l], 1)
                    ).T
                    == policy_logit_[l]
                )
                prob = (1 - epsilon) * greedy_flg + epsilon / self.n_unique_action[l]
                pscore_all.append(prob)

        return pscore_all

    def calc_on_policy_policy_value(
        self,
        label: np.ndarray,
        policy_logit_: List[np.ndarray],
        epsilon: Optional[float] = None,
    ) -> float:
        """Calculate the policy value of given reward and slate_id.

        Parameters
        -----------
        policy_logit_: list of array-like, (n_rounds, n_unique_actions[l])
            Logit values to define the evaluation policy.

        epsilon: float, default=None
            Exploration rate of the epsilon-greedy policy.
            When `None` is given, the softmax policy is used instead.

        Returns
        ----------
        policy_value: float
            On-policy policy value estimate of the behavior policy.

        """
        action, _ = self.sample_action_and_obtain_pscore(
            policy_logit_=policy_logit_,
            epsilon=epsilon,
        )
        expected_reward_factual = self.reward_function(
            action=action,
            label=label,
            candidate_actions=self.candidate_actions,
            random_state=self.base_random_state,
        )
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual
        )
        return reward.mean()


def base_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
):
    n_rounds, slate_size = action.shape
    random_ = check_random_state(random_state)

    action_coef_all = []
    for l in range(slate_size):
        action_coef_all.append(
            random_.uniform(0.0, 0.5, size=len(candidate_actions[l]))
        )

    positive_flg = np.zeros((n_rounds, slate_size))
    action_coef = np.zeros((n_rounds, slate_size))
    for l in range(slate_size):
        positive_flg[:, l] = label[
            np.arange(n_rounds), candidate_actions[l][action[:, l]]
        ]
        action_coef[:, l] = action_coef_all[l][action[:, l]]

    base_expected_reward_factual = (1 - action_coef) * positive_flg + action_coef
    return base_expected_reward_factual


def clipped_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
    **kwargs,
):
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )
    slate_size = action.shape[1]
    base_expected_reward_factual = base_expected_reward_factual[
        :, : slate_size // 2
    ].mean(axis=1)

    expected_reward_factual = np.clip(base_expected_reward_factual, 0.0, 1.0)
    return expected_reward_factual


def minmax_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
    **kwargs,
):
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )

    slate_size = action.shape[1]
    expected_reward_factual = base_expected_reward_factual[:, : slate_size // 2].max(
        axis=1
    ) + base_expected_reward_factual[:, : slate_size // 2].min(axis=1)
    return expected_reward_factual


def discretized_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
    **kwargs,
):
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )

    slate_size = action.shape[1]
    base_expected_reward_factual = base_expected_reward_factual[
        :, : slate_size // 2
    ].mean(axis=1)

    expected_reward_factual = np.floor(base_expected_reward_factual * 4) / 4
    return expected_reward_factual


def logistic_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
    **kwargs,
):
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )

    slate_size = action.shape[1]
    base_expected_reward_factual = base_expected_reward_factual[
        :, : slate_size // 2
    ].mean(axis=1)

    expected_reward_factual = sigmoid((base_expected_reward_factual - 0.5) * 5.0)
    return expected_reward_factual


def additive_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
    **kwargs,
):
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )

    slate_size = action.shape[1]
    base_expected_reward_factual = base_expected_reward_factual[
        :, : slate_size // 2
    ].mean(axis=1)

    n_unique_actions = np.zeros(slate_size // 2, dtype=int)
    for l in range(slate_size // 2):
        n_unique_actions[l] = candidate_actions[l].shape[0]

    random_ = check_random_state(random_state)

    interaction_matrix = []
    for l in range(slate_size // 2 - 1):
        interaction_matrix.append(
            random_.normal(size=(n_unique_actions[l], n_unique_actions[l + 1]))
        )

    for l in range(slate_size // 2 - 1):
        base_expected_reward_factual += (
            interaction_matrix[l][action[:, l], action[:, l + 1]]
        ) / (slate_size - 1)

    expected_reward_factual = base_expected_reward_factual
    return expected_reward_factual


def product_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
    **kwargs,
):
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )

    slate_size = action.shape[1]
    expected_reward_factual = np.power(
        np.abs(base_expected_reward_factual[:, : slate_size // 2].prod(axis=1)),
        1 / (slate_size // 2),
    )
    return expected_reward_factual


def discount_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
):
    slate_size = action.shape[1]
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )[:, : slate_size // 2]

    random_ = check_random_state(random_state)

    n_unique_actions = np.zeros(slate_size // 2, dtype=int)
    for l in range(slate_size // 2):
        n_unique_actions[l] = candidate_actions[l].shape[0]

    # discount
    interaction_matrix = []
    for l in range(slate_size // 2 - 1):
        interaction_matrix.append(
            random_.normal(size=(n_unique_actions[l], n_unique_actions[l + 1]))
        )

    for l in range(slate_size // 2 - 1):
        base_expected_reward_factual[:, l + 1] *= interaction_matrix[l][
            action[:, l], action[:, l + 1]
        ]

    expected_reward_factual = base_expected_reward_factual.mean(axis=1)
    return expected_reward_factual


def sparse_reward_function(
    label: np.ndarray,
    action: np.ndarray,
    candidate_actions: List[np.ndarray],
    random_state: int,
):
    slate_size = action.shape[1]
    base_expected_reward_factual = base_reward_function(
        label=label,
        action=action,
        candidate_actions=candidate_actions,
        random_state=random_state,
    )[:, : slate_size // 2]

    random_ = check_random_state(random_state)

    n_unique_actions = np.zeros(slate_size // 2, dtype=int)
    for l in range(slate_size // 2):
        n_unique_actions[l] = candidate_actions[l].shape[0]

    # discount
    interaction_matrix = []
    for l in range(slate_size // 2 - 1):
        interaction_matrix.append(
            random_.normal(size=(n_unique_actions[l], n_unique_actions[l + 1]))
        )

    for l in range(slate_size // 2 - 1):
        base_expected_reward_factual[:, l + 1] *= interaction_matrix[l][
            action[:, l], action[:, l + 1]
        ]

    expected_reward_factual = base_expected_reward_factual.mean(axis=1)

    # expected_reward_factual = np.sqrt(1 - (base_expected_reward_factual - 1) ** 2)

    # expected_reward_factual = np.zeros((len(base_expected_reward_factual),))
    # for l in range(slate_size // 2 - 1):
    #     expected_reward_factual += (
    #         base_expected_reward_factual[:, l] * base_expected_reward_factual[:, l + 1]
    #     )

    # slate_size = action.shape[1]
    # base_expected_reward_factual = base_expected_reward_factual[
    #     :, : slate_size // 2
    # ].mean(axis=1)
    # expected_reward_factual = (
    #     base_expected_reward_factual
    #     + np.abs(np.sin(32 * np.pi * base_expected_reward_factual))
    # ) / 2

    # poly = PolynomialFeatures(degree=(2, 3))
    # base_expected_reward_factual = poly.fit_transform(base_expected_reward_factual)

    # random_ = check_random_state(random_state)
    # weight = random_.uniform(size=(base_expected_reward_factual.shape[1],)).reshape(1, -1)
    # weight = np.tile(weight, (len(base_expected_reward_factual), 1))

    # expected_reward_factual = (weight * base_expected_reward_factual).mean(axis=1)

    # slate_size = action.shape[1]
    # base_expected_reward_factual = base_expected_reward_factual[:, : slate_size // 2]

    # n_classes = 10
    # coef = random_.uniform((slate_size, n_classes))

    # slate_size = action.shape[1]
    # base_expected_reward_factual = base_expected_reward_factual[
    #     :, : slate_size // 2
    # ].mean(axis=1)

    # expected_reward_factual = (
    #     base_expected_reward_factual * 5 - np.floor(base_expected_reward_factual * 5)
    # ) / 5

    return expected_reward_factual
