"""param search"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any
import random

Space = Dict[str, tuple]
Objective = Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]]


def _has_optuna() -> bool:
    try:
        import optuna  # noqa
        return True
    except Exception:
        return False


def run_search(objective: Objective, space: Space, n_trials: int, seed: int = 0,
               early_stop: bool = False, early_stop_margin: float = 0.01,
               adaptive: bool = False):
    """run search"""
    actual_trials = n_trials
    if adaptive:
        actual_trials = min(6, n_trials)  # start small
    history: List[dict] = []

    if _has_optuna():
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _obj(trial):
            params = {}
            for name, spec in space.items():
                kind = spec[0]
                if kind == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif kind == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif kind == "cat":
                    params[name] = trial.suggest_categorical(name, spec[1])

            print(f"\n[Trial {len(history) + 1}/{actual_trials}]")
            print(f"  Parameters: {params}")

            value, attrs = objective(params)

            print(f"  Metrics: {attrs}")
            print(f"  Loss: {value:.6f}")

            for k, v in attrs.items():
                trial.set_user_attr(k, v)
            history.append({"params": params, "value": value, **attrs})

            if early_stop and len(history) >= 5:
                sorted_h = sorted(history, key=lambda h: h["value"])[:3]
                best_v = sorted_h[0]["value"]
                worst_v = sorted_h[2]["value"]
                if worst_v - best_v <= early_stop_margin * best_v and len(history) >= 8:
                    print("  → Early stopping triggered")
                    raise optuna.TrialPruned()
            return value

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(_obj, n_trials=actual_trials, show_progress_bar=False)

        if adaptive and actual_trials == 6 and len(history) == 6:
            study.optimize(_obj, n_trials=n_trials - 6, show_progress_bar=False)

        return study.best_params, study.best_trial.user_attrs, study.best_value, history

    rng = random.Random(seed)
    best = None
    for trial_num in range(n_trials):
        params = {}
        for name, spec in space.items():
            kind = spec[0]
            if kind == "float":
                params[name] = rng.uniform(spec[1], spec[2])
            elif kind == "int":
                params[name] = rng.randint(spec[1], spec[2])
            elif kind == "cat":
                params[name] = rng.choice(spec[1])

        print(f"\n[Trial {trial_num + 1}/{n_trials}]")
        print(f"  Parameters: {params}")

        value, attrs = objective(params)

        print(f"  Metrics: {attrs}")
        print(f"  Loss: {value:.6f}")

        history.append({"params": params, "value": value, **attrs})
        if best is None or value < best[2]:
            best = (params, attrs, value)
            print(f"  ✓ New best!")

    return best[0], best[1], best[2], history
