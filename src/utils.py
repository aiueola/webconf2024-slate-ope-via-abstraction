from typing import Optional, Dict

import torch
import numpy as np
from sklearn.utils import check_scalar, check_random_state


def cosine_similarity(x, y):
    return x.T @ y / (np.linalg.norm(x) * np.linalg.norm(y))


def to_tensor(
    arr: np.ndarray,
    dtype: type = float,
    device: str = "cuda:0",
):
    tensor = torch.FloatTensor(arr) if dtype == float else torch.LongTensor(arr)
    return tensor.to(device)


def torch_seed(random_state: int, device=str):
    if device == "cuda:0":
        torch.cuda.manual_seed(random_state)

    torch.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_confidence_interval_arguments(
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Optional[ValueError]:
    """Check confidence interval arguments.
    Parameters
    ----------
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
    check_random_state(random_state)
    check_scalar(alpha, "alpha", float, min_val=0.0, max_val=1.0)
    check_scalar(n_bootstrap_samples, "n_bootstrap_samples", int, min_val=1)


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval using bootstrap.
    Parameters
    ----------
    samples: array-like
        Empirical observed samples to be used to estimate cumulative distribution function.
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
    check_confidence_interval_arguments(
        alpha=alpha, n_bootstrap_samples=n_bootstrap_samples, random_state=random_state
    )

    boot_samples = list()
    random_ = check_random_state(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
    }


def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
    expected_dtype: Optional[type] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> ValueError:
    """Input validation on array.

    Parameters
    -------
    array: object
        Input array to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Expected dimension of the input array.

    expected_dtype: {type, tuple of type}, default=None
        Expected dtype of the input array.

    min_val: float, default=None
        Minimum number allowed in the input array.

    max_val: float, default=None
        Maximum number allowed in the input array.

    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be {expected_dim}D array, but got {type(array)}")
    if array.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D array, but got {array.ndim}D array"
        )
    if expected_dtype is not None:
        if not np.issubsctype(array, expected_dtype):
            raise ValueError(
                f"The elements of {name} must be {expected_dtype}, but got {array.dtype}"
            )
    if min_val is not None:
        if array.min() < min_val:
            raise ValueError(
                f"The elements of {name} must be larger than {min_val}, but got minimum value {array.min()}"
            )
    if max_val is not None:
        if array.max() > max_val:
            raise ValueError(
                f"The elements of {name} must be smaller than {max_val}, but got maximum value {array.max()}"
            )


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"
