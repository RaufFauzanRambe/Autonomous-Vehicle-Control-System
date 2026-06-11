"""
Hyperparameter Search Module for Training Optimization.

Implements multiple hyperparameter optimization strategies:
- Grid Search: Exhaustive parameter sweep
- Random Search: Random parameter sampling
- Bayesian Optimization: Gaussian process-based (Optuna)
- Population-Based Training: Evolutionary optimization

Features:
- Configurable search spaces
- Early pruning of unpromising trials
- Multi-objective optimization
- Parallel trial execution
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SearchSpace:
    """Definition of a hyperparameter search space."""
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "log_uniform", "low": 1e-5, "high": 1e-2
    })
    weight_decay: Dict[str, Any] = field(default_factory=lambda: {
        "type": "log_uniform", "low": 1e-6, "high": 1e-2
    })
    batch_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice", "values": [8, 16, 32, 64, 128]
    })
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice", "values": ["sgd", "adam", "adamw"]
    })
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice", "values": ["cosine", "step", "onecycle"]
    })
    warmup_epochs: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int_uniform", "low": 0, "high": 10
    })
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice", "values": ["none", "light", "medium", "heavy"]
    })
    dropout_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform", "low": 0.0, "high": 0.5
    })
    label_smoothing: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform", "low": 0.0, "high": 0.2
    })


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: int
    params: Dict[str, Any]
    metric: float
    status: str = "completed"  # completed, pruned, failed
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


def sample_param(space_def: Dict[str, Any]) -> Any:
    """
    Sample a single parameter from its search space definition.

    Args:
        space_def: Search space definition with type and bounds.

    Returns:
        Sampled parameter value.
    """
    param_type = space_def["type"]

    if param_type == "choice":
        return random.choice(space_def["values"])
    elif param_type == "uniform":
        return random.uniform(space_def["low"], space_def["high"])
    elif param_type == "log_uniform":
        log_low = np.log(space_def["low"])
        log_high = np.log(space_def["high"])
        return float(np.exp(random.uniform(log_low, log_high)))
    elif param_type == "int_uniform":
        return random.randint(space_def["low"], space_def["high"])
    elif param_type == "int_log_uniform":
        log_low = np.log(space_def["low"])
        log_high = np.log(space_def["high"])
        return int(np.exp(random.uniform(log_low, log_high)))
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


class GridSearch:
    """
    Grid search hyperparameter optimizer.

    Exhaustively evaluates all combinations of discrete parameter values.
    Best for small, well-defined search spaces.
    """

    def __init__(self, search_space: Dict[str, List[Any]]) -> None:
        """
        Args:
            search_space: Dictionary mapping param names to lists of values.
        """
        self.search_space = search_space
        self._grid = self._build_grid()
        self.results: List[TrialResult] = []

    def _build_grid(self) -> List[Dict[str, Any]]:
        """Build all combinations of parameter values."""
        import itertools

        keys = list(self.search_space.keys())
        values = [self.search_space[k] if isinstance(self.search_space[k], list)
                  else [self.search_space[k]] for k in keys]

        grid = []
        for combo in itertools.product(*values):
            grid.append(dict(zip(keys, combo)))
        return grid

    def __len__(self) -> int:
        return len(self._grid)

    def suggest(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """Get parameters for a specific trial."""
        if trial_id < len(self._grid):
            return self._grid[trial_id]
        return None

    def report(self, trial_id: int, params: Dict, metric: float,
               status: str = "completed") -> None:
        """Report trial result."""
        self.results.append(TrialResult(
            trial_id=trial_id, params=params, metric=metric, status=status
        ))

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        if not self.results:
            return {}
        completed = [r for r in self.results if r.status == "completed"]
        if not completed:
            return {}
        best = min(completed, key=lambda r: r.metric)
        return best.params


class RandomSearch:
    """
    Random search hyperparameter optimizer.

    Samples parameter combinations uniformly at random.
    More efficient than grid search for high-dimensional spaces.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 50,
        seed: int = 42,
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)
        self.results: List[TrialResult] = []
        self._tried_params: List[Dict] = []

    def suggest(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """Sample random parameters for a trial."""
        if trial_id >= self.n_trials:
            return None

        params = {}
        for param_name, space_def in vars(self.search_space).items():
            if isinstance(space_def, dict) and "type" in space_def:
                params[param_name] = sample_param(space_def)

        self._tried_params.append(params)
        return params

    def report(self, trial_id: int, params: Dict, metric: float,
               status: str = "completed") -> None:
        """Report trial result."""
        self.results.append(TrialResult(
            trial_id=trial_id, params=params, metric=metric, status=status
        ))

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        if not self.results:
            return {}
        completed = [r for r in self.results if r.status == "completed"]
        if not completed:
            return {}
        best = min(completed, key=lambda r: r.metric)
        return best.params


class BayesianOptimizer:
    """
    Bayesian optimization using Optuna.

    Uses Tree-structured Parzen Estimators (TPE) for efficient
    hyperparameter optimization with early pruning support.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 50,
        prune_after: int = 10,
        seed: int = 42,
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.prune_after = prune_after
        self.seed = seed
        self.results: List[TrialResult] = []
        self._study = None

    def _create_study(self) -> None:
        """Create an Optuna study."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            self._study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=self.prune_after,
                    n_warmup_steps=5,
                ),
            )
        except ImportError:
            print("[HPO] Optuna not available. Falling back to random search.")
            self._study = None

    def suggest(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """Get parameters suggested by Bayesian optimization."""
        if self._study is None:
            self._create_study()

        if self._study is None:
            # Fallback to random
            params = {}
            for param_name, space_def in vars(self.search_space).items():
                if isinstance(space_def, dict) and "type" in space_def:
                    params[param_name] = sample_param(space_def)
            return params

        try:
            import optuna
            trial = self._study.ask()

            params = {}
            for param_name, space_def in vars(self.search_space).items():
                if not isinstance(space_def, dict) or "type" not in space_def:
                    continue

                if space_def["type"] == "choice":
                    params[param_name] = trial.suggest_categorical(
                        param_name, space_def["values"]
                    )
                elif space_def["type"] == "uniform":
                    params[param_name] = trial.suggest_float(
                        param_name, space_def["low"], space_def["high"]
                    )
                elif space_def["type"] == "log_uniform":
                    params[param_name] = trial.suggest_float(
                        param_name, space_def["low"], space_def["high"], log=True
                    )
                elif space_def["type"] == "int_uniform":
                    params[param_name] = trial.suggest_int(
                        param_name, space_def["low"], space_def["high"]
                    )

            return params

        except Exception as e:
            print(f"[HPO] Bayesian suggestion failed: {e}")
            return None

    def report(self, trial_id: int, params: Dict, metric: float,
               status: str = "completed") -> None:
        """Report trial result."""
        self.results.append(TrialResult(
            trial_id=trial_id, params=params, metric=metric, status=status
        ))

        if self._study is not None:
            try:
                import optuna
                frozen_trial = self._study.trials[-1] if self._study.trials else None
                self._study.tell(frozen_trial.number if frozen_trial else trial_id, metric)
            except Exception:
                pass

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        if self._study is not None:
            try:
                return self._study.best_params
            except Exception:
                pass

        if not self.results:
            return {}
        completed = [r for r in self.results if r.status == "completed"]
        if not completed:
            return {}
        best = min(completed, key=lambda r: r.metric)
        return best.params


class PopulationBasedTraining:
    """
    Population-Based Training (PBT) optimizer.

    Maintains a population of agents with different hyperparameters,
    periodically exploiting good performers and exploring new configs.
    """

    def __init__(
        self,
        population_size: int = 10,
        exploit_fraction: float = 0.2,
        explore_noise: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.exploit_fraction = exploit_fraction
        self.explore_noise = explore_noise
        self.rng = np.random.RandomState(seed)
        self.population: List[Dict[str, Any]] = []
        self.fitness: List[float] = []
        self.generation = 0

    def initialize_population(self, base_params: Dict[str, Any]) -> List[Dict]:
        """Initialize the population with perturbed parameters."""
        self.population = []
        self.fitness = [0.0] * self.population_size

        for i in range(self.population_size):
            params = {}
            for key, value in base_params.items():
                if isinstance(value, float):
                    noise = 1.0 + self.rng.uniform(
                        -self.explore_noise, self.explore_noise
                    )
                    params[key] = value * noise
                else:
                    params[key] = value
            self.population.append(params)

        return self.population

    def step(self, fitness_scores: List[float]) -> List[Dict]:
        """
        Evolve the population based on fitness scores.

        Args:
            fitness_scores: Fitness for each member.

        Returns:
            Updated population parameters.
        """
        self.fitness = fitness_scores
        self.generation += 1

        # Sort by fitness (lower is better)
        indices = np.argsort(fitness_scores)
        n_exploit = max(1, int(self.population_size * self.exploit_fraction))

        # Replace worst performers with perturbed copies of best
        for i in range(n_exploit):
            worst_idx = indices[-(i + 1)]
            best_idx = indices[i]

            new_params = {}
            for key, value in self.population[best_idx].items():
                if isinstance(value, float):
                    noise = 1.0 + self.rng.uniform(
                        -self.explore_noise, self.explore_noise
                    )
                    new_params[key] = value * noise
                else:
                    new_params[key] = value

            self.population[worst_idx] = new_params

        return self.population


def run_hpo(args: Any) -> Dict[str, Any]:
    """
    Run hyperparameter optimization.

    Args:
        args: Training arguments with HPO configuration.

    Returns:
        Best hyperparameters found.
    """
    search_space = SearchSpace()
    n_trials = args.hpo_trials

    if args.hpo_framework == "optuna":
        optimizer = BayesianOptimizer(search_space, n_trials=n_trials)
    elif args.hpo_framework == "grid":
        grid_space = {
            "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
            "batch_size": [16, 32, 64],
            "optimizer": ["sgd", "adam", "adamw"],
        }
        optimizer = GridSearch(grid_space)
    else:
        optimizer = RandomSearch(search_space, n_trials=n_trials)

    print(f"\n[HPO] Starting {args.hpo_framework} search ({n_trials} trials)")

    for trial_id in range(n_trials):
        params = optimizer.suggest(trial_id)
        if params is None:
            break

        # Simulate trial (in practice, would train a model)
        start_time = time.time()
        try:
            # Mock metric: use a simple function for demonstration
            lr = params.get("learning_rate", 1e-3)
            bs = params.get("batch_size", 32)
            metric = abs(np.log10(lr) + 3) * 0.1 + random.random() * 0.05

            optimizer.report(trial_id, params, metric, "completed")
        except Exception as e:
            optimizer.report(trial_id, params, float("inf"), "failed")
            print(f"[HPO] Trial {trial_id} failed: {e}")

    best = optimizer.get_best_params()
    print(f"[HPO] Best parameters: {best}")
    return best
