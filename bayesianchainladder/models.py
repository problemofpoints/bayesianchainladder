"""
Low-level model building functions for Bayesian chain ladder.

This module provides functions to build Bambi and PyMC models from
triangle data and formulas for stochastic loss reserving.
"""

from __future__ import annotations

from typing import Any, Literal

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr


def build_bambi_model(
    data: pd.DataFrame,
    formula: str = "incremental ~ 1 + C(origin) + C(dev)",
    family: str = "negativebinomial",
    link: str | None = None,
    priors: dict[str, Any] | None = None,
    offset: str | pd.Series | np.ndarray | None = None,
    potentials: list[tuple] | None = None,
) -> bmb.Model:
    """
    Build a Bambi model for chain ladder GLM.

    This function creates a Bambi model suitable for fitting a cross-classified
    chain ladder model to loss triangle data.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame with columns for response and predictors.
        Must include columns matching those in the formula.
    formula : str, optional
        Bambi/Patsy-style formula specifying the model.
        Default is "incremental ~ 1 + C(origin) + C(dev)" for the standard
        cross-classified chain ladder model.
    family : str, optional
        The response distribution family. Options include:
        - "negativebinomial": Negative binomial (overdispersed counts)
        - "poisson": Poisson (for count data)
        - "gamma": Gamma (for positive continuous data)
        - "gaussian": Normal/Gaussian
        - "wald" / "inverse_gaussian": Wald / inverse-Gaussian (heavy-tailed positive data)
        Default is "negativebinomial" as an overdispersed Poisson proxy.
    link : str, optional
        Link function. If None, uses the default for the family.
        Common options: "log", "identity".
    priors : dict, optional
        Dictionary of prior specifications for model parameters.
        Keys are parameter names, values are bambi.Prior objects or dicts.
    offset : str or array-like, optional
        Offset term for the model (e.g., log-exposure).
        If str, should be a column name in data.
    potentials : list of 2-tuples, optional
        Arbitrary expressions added to the log-likelihood.  Each tuple is
        ``(variable_name_or_tuple, constraint_fn)`` — the variable name(s)
        are looked up in the PyMC model and passed to the constraint function.
        See Bambi docs for details.  Primary use: zero-weight dummy rows by
        passing a potential that subtracts their per-observation log-likelihood
        contribution.

    Returns
    -------
    bmb.Model
        A Bambi model object ready for fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from bayesianchainladder.models import build_bambi_model
    >>> data = pd.DataFrame({
    ...     "incremental": [100, 80, 60, 110, 90, 120],
    ...     "origin": [1, 1, 1, 2, 2, 3],
    ...     "dev": [1, 2, 3, 1, 2, 1],
    ... })
    >>> model = build_bambi_model(data)
    """
    # Process offset - in Bambi, offset is added to data and included via formula
    model_data = data.copy()

    if offset is not None:
        if isinstance(offset, str):
            if offset in data.columns:
                offset_values = np.asarray(data[offset].values, dtype=np.float64)
                model_data["logoffset"] = np.log(offset_values)
            else:
                raise ValueError(f"Offset column '{offset}' not found in data")
        else:
            offset_term = np.asarray(offset, dtype=np.float64)
            if len(offset_term) != len(data):
                raise ValueError("Offset length must match data length")
            model_data["logoffset"] = offset_term

        # Add offset to formula using Bambi's offset() function
        formula = formula + " + offset(logoffset)"

    # Build family specification
    family_spec = _get_family(family, link)

    # Create the model
    model = bmb.Model(
        formula=formula,
        data=model_data,
        family=family_spec,
        priors=priors,
        potentials=potentials,
    )

    return model


def _get_family(family: str, link: str | None = None) -> "str | bmb.Family":
    """Return a Bambi family spec — either a name string (default link) or a custom Family object.

    When ``link`` matches the family default (or is None), a plain string is returned so that
    Bambi uses its built-in prior / link configuration unchanged.  When a non-default link is
    requested, a fully-constructed ``bmb.Family`` object is returned so that Bambi honours the
    requested link instead of silently falling back to its default.

    The most important non-default case for actuarial work is ``family="gamma", link="log"``.
    Bambi's default gamma link is ``inverse``; explicitly requesting ``log`` gives the
    log-linear chain-ladder model that the docstring describes.

    Parameters
    ----------
    family : str
        Family name.  Aliases accepted: ``"negative_binomial"`` / ``"negbinom"`` for
        ``"negativebinomial"``; ``"normal"`` for ``"gaussian"``.
    link : str or None
        Link function name (``"log"``, ``"identity"``, ``"inverse"``, …).  ``None`` means use
        the family default.

    Returns
    -------
    str or bmb.Family
        Plain string when the default link is adequate; ``bmb.Family`` instance otherwise.
    """
    from bambi.defaults.utils import generate_family as _bmb_gen_family

    # Map aliases to canonical Bambi family names.
    family_map = {
        "negativebinomial": "negativebinomial",
        "negative_binomial": "negativebinomial",
        "negbinom": "negativebinomial",
        "poisson": "poisson",
        "gamma": "gamma",
        "gaussian": "gaussian",
        "normal": "gaussian",
        "wald": "wald",
        "inverse_gaussian": "wald",
        "inversegaussian": "wald",
        "t": "t",
        "student_t": "t",
        "studentt": "t",
    }

    family_lower = family.lower()
    if family_lower not in family_map:
        raise ValueError(
            f"Unknown family '{family}'. Supported families: {list(family_map.keys())}"
        )

    bambi_name = family_map[family_lower]

    # Use the fast path (plain string) when the link is the family default.
    if link is None or link == _get_default_link(bambi_name):
        return bambi_name

    # Non-default link: construct an explicit Family object so Bambi honours it.
    # Each entry mirrors the BUILTIN_FAMILIES spec in bambi/defaults/families.py,
    # but with the mu link replaced by the caller's choice.
    # Non-parent auxiliary parameters (alpha, sigma, …) keep their canonical log links
    # so Bambi's auto-generated priors remain valid.
    _family_specs: dict[str, dict] = {
        "gamma": {
            "likelihood": {"name": "Gamma", "params": ["mu", "alpha"], "parent": "mu"},
            "link": {"mu": link, "alpha": "log"},
            "family_cls_name": "Gamma",
            "default_priors": {"alpha": "HalfCauchy"},
        },
        "negativebinomial": {
            "likelihood": {"name": "NegativeBinomial", "params": ["mu", "alpha"], "parent": "mu"},
            "link": {"mu": link, "alpha": "log"},
            "family_cls_name": "NegativeBinomial",
            "default_priors": {"alpha": "HalfCauchy"},
        },
        "poisson": {
            "likelihood": {"name": "Poisson", "params": ["mu"], "parent": "mu"},
            "link": {"mu": link},
            "family_cls_name": "Poisson",
            "default_priors": {},
        },
        "gaussian": {
            "likelihood": {"name": "Normal", "params": ["mu", "sigma"], "parent": "mu"},
            "link": {"mu": link, "sigma": "log"},
            "family_cls_name": "Gaussian",
            "default_priors": {"sigma": "HalfNormal"},
        },
        "wald": {
            "likelihood": {"name": "Wald", "params": ["mu", "lam"], "parent": "mu"},
            "link": {"mu": link, "lam": "log"},
            "family_cls_name": "Wald",
            "default_priors": {"lam": "HalfCauchy"},
        },
        "t": {
            "likelihood": {"name": "StudentT", "params": ["mu", "sigma", "nu"], "parent": "mu"},
            "link": {"mu": link, "sigma": "log", "nu": "log"},
            "family_cls_name": "StudentT",
            "default_priors": {"sigma": "HalfNormal", "nu": "Gamma"},
        },
    }

    if bambi_name not in _family_specs:
        # Fallback for any future family additions — warn and return the string.
        import warnings
        warnings.warn(
            f"Custom link '{link}' for family '{bambi_name}' is not supported; "
            "using the default link instead.",
            UserWarning,
        )
        return bambi_name

    spec = _family_specs[bambi_name]

    # Dynamically import the Bambi family class by name.
    from bambi.families import univariate as _bmb_univariate
    family_cls = getattr(_bmb_univariate, spec["family_cls_name"])

    return _bmb_gen_family(
        name=bambi_name,
        likelihood=spec["likelihood"],
        link=spec["link"],
        family=family_cls,
        default_priors=spec["default_priors"] if spec["default_priors"] else None,
    )


def _get_default_link(family: str) -> str:
    """Get the default mu-link function for a Bambi family."""
    default_links = {
        "negativebinomial": "log",
        "poisson": "log",
        "gamma": "inverse",
        "gaussian": "identity",
        "wald": "inverse_squared",
        "t": "identity",
    }
    return default_links.get(family, "identity")


def build_pymc_model(
    data: pd.DataFrame,
    response_col: str = "incremental",
    origin_col: str = "origin",
    dev_col: str = "dev",
    calendar_col: str | None = None,
    family: str = "negativebinomial",
    include_intercept: bool = True,
    exposure_col: str | None = None,
    priors: dict[str, Any] | None = None,
) -> pm.Model:
    """
    Build a PyMC model for chain ladder GLM.

    This function creates a PyMC model directly, giving more control
    over the model structure than the Bambi wrapper.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame with triangle data.
    response_col : str, optional
        Name of the response column. Default is "incremental".
    origin_col : str, optional
        Name of the origin period column. Default is "origin".
    dev_col : str, optional
        Name of the development period column. Default is "dev".
    calendar_col : str, optional
        Name of calendar period column. If None, calendar effects not included.
    family : str, optional
        Response distribution family. Default is "negativebinomial".
    include_intercept : bool, optional
        Whether to include an intercept term. Default is True.
    exposure_col : str, optional
        Name of exposure column for offset term.
    priors : dict, optional
        Custom prior specifications.

    Returns
    -------
    pm.Model
        A PyMC model object.
    """
    priors = priors or {}

    # Get data arrays
    y = data[response_col].values
    n_obs = len(y)

    # Encode categorical variables
    origin_codes, origin_levels = pd.factorize(data[origin_col], sort=True)
    dev_codes, dev_levels = pd.factorize(data[dev_col], sort=True)

    n_origin = len(origin_levels)
    n_dev = len(dev_levels)

    coords = {
        "origin": origin_levels,
        "dev": dev_levels,
        "obs": np.arange(n_obs),
    }

    # Handle calendar effects if included
    if calendar_col is not None:
        calendar_codes, calendar_levels = pd.factorize(data[calendar_col], sort=True)
        n_calendar = len(calendar_levels)
        coords["calendar"] = calendar_levels

    # Handle exposure offset
    log_exposure = None
    if exposure_col is not None:
        exposure = np.asarray(data[exposure_col].values, dtype=np.float64)
        log_exposure = np.log(exposure)

    with pm.Model(coords=coords) as model:
        # Data containers
        origin_idx = pm.Data("origin_idx", origin_codes, dims="obs")
        dev_idx = pm.Data("dev_idx", dev_codes, dims="obs")

        if calendar_col is not None:
            calendar_idx = pm.Data("calendar_idx", calendar_codes, dims="obs")

        # Priors for origin effects
        origin_prior = priors.get("origin", {"sigma": 1.0})
        origin_sigma = origin_prior.get("sigma", 1.0)
        alpha_origin = pm.Normal(
            "alpha_origin", mu=0, sigma=origin_sigma, dims="origin"
        )

        # Priors for development effects
        dev_prior = priors.get("dev", {"sigma": 1.0})
        dev_sigma = dev_prior.get("sigma", 1.0)
        alpha_dev = pm.Normal("alpha_dev", mu=0, sigma=dev_sigma, dims="dev")

        # Build linear predictor
        if include_intercept:
            intercept_prior = priors.get("intercept", {"mu": 0, "sigma": 10})
            intercept = pm.Normal(
                "intercept",
                mu=intercept_prior.get("mu", 0),
                sigma=intercept_prior.get("sigma", 10),
            )
            mu = intercept + alpha_origin[origin_idx] + alpha_dev[dev_idx]
        else:
            mu = alpha_origin[origin_idx] + alpha_dev[dev_idx]

        # Add calendar effects if specified
        if calendar_col is not None:
            calendar_prior = priors.get("calendar", {"sigma": 0.5})
            calendar_sigma = calendar_prior.get("sigma", 0.5)
            alpha_calendar = pm.Normal(
                "alpha_calendar", mu=0, sigma=calendar_sigma, dims="calendar"
            )
            mu = mu + alpha_calendar[calendar_idx]

        # Add exposure offset
        if log_exposure is not None:
            offset = pm.Data("log_exposure", log_exposure, dims="obs")
            mu = mu + offset

        # Response distribution
        if family.lower() in ("negativebinomial", "negative_binomial", "negbinom"):
            # Dispersion parameter
            alpha_prior = priors.get("alpha", {"alpha": 2, "beta": 1})
            alpha = pm.Gamma(
                "alpha",
                alpha=alpha_prior.get("alpha", 2),
                beta=alpha_prior.get("beta", 1),
            )
            pm.NegativeBinomial("y", mu=pt.exp(mu), alpha=alpha, observed=y)

        elif family.lower() == "poisson":
            pm.Poisson("y", mu=pt.exp(mu), observed=y)

        elif family.lower() == "gamma":
            sigma_prior = priors.get("sigma", {"sigma": 1})
            sigma = pm.HalfNormal("sigma", sigma=sigma_prior.get("sigma", 1))
            pm.Gamma(
                "y",
                alpha=pt.exp(mu) / sigma,
                beta=1 / sigma,
                observed=y,
            )

        elif family.lower() in ("gaussian", "normal"):
            sigma_prior = priors.get("sigma", {"sigma": 1})
            sigma = pm.HalfNormal("sigma", sigma=sigma_prior.get("sigma", 1))
            pm.Normal("y", mu=pt.exp(mu), sigma=sigma, observed=y)

        else:
            raise ValueError(f"Unknown family: {family}")

    return model


def fit_model(
    model: bmb.Model | pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = None,
    **kwargs: Any,
) -> az.InferenceData:
    """
    Fit a Bambi or PyMC model using MCMC.

    Parameters
    ----------
    model : bmb.Model or pm.Model
        The model to fit.
    draws : int, optional
        Number of posterior samples per chain. Default is 2000.
    tune : int, optional
        Number of tuning samples. Default is 1000.
    chains : int, optional
        Number of MCMC chains. Default is 4.
    target_accept : float, optional
        Target acceptance probability for NUTS. Default is 0.9.
    random_seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to the sampler.

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object with posterior samples.
    """
    if isinstance(model, bmb.Model):
        # Use adaptive init for better starting points
        init_kwargs = kwargs.pop("init", None)
        if init_kwargs is None:
            init_kwargs = "adapt_diag"

        # Ensure log_likelihood is computed for model comparison (LOO, WAIC)
        idata_kwargs = kwargs.pop("idata_kwargs", {})
        if "log_likelihood" not in idata_kwargs:
            idata_kwargs["log_likelihood"] = True

        idata = model.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            init=init_kwargs,
            idata_kwargs=idata_kwargs,
            **kwargs,
        )
    else:
        # PyMC model
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
                **kwargs,
            )

    return idata


def predict_posterior(
    model: bmb.Model | pm.Model,
    idata: az.InferenceData,
    data: pd.DataFrame,
    kind: Literal["mean", "response"] = "mean",
    include_group_specific: bool = True,
    origin_col: str = "origin",
    dev_col: str = "dev",
    calendar_col: str | None = None,
    exposure_col: str | None = None,
) -> xr.DataArray:
    """
    Generate posterior predictions for new data.

    Parameters
    ----------
    model : bmb.Model or pm.Model
        The fitted Bambi or PyMC model.
    idata : az.InferenceData
        The inference data from model fitting.
    data : pd.DataFrame
        New data for predictions (must have same columns as training data).
    kind : {"mean", "response"}, optional
        Type of prediction:
        - "mean": Predicted mean (expected value)
        - "response": Predicted responses (samples from posterior predictive)
        Default is "mean".
    include_group_specific : bool, optional
        Whether to include group-specific effects in predictions.
        Only used for Bambi models. Default is True.
    origin_col : str, optional
        Name of origin column. Only used for PyMC models. Default is "origin".
    dev_col : str, optional
        Name of development column. Only used for PyMC models. Default is "dev".
    calendar_col : str, optional
        Name of calendar column. Only used for PyMC models. Default is None.
    exposure_col : str, optional
        Name of exposure column. Only used for PyMC models. Default is None.

    Returns
    -------
    xr.DataArray
        DataArray with posterior predictions.
    """
    if isinstance(model, bmb.Model):
        # Use Bambi's predict method
        model.predict(
            idata,
            data=data,
            kind=kind,
            inplace=True,
            include_group_specific=include_group_specific,
        )

        if kind == "mean":
            return idata.posterior[f"{model.response_component.response.name}_mean"] # type: ignore
        else:
            return idata.posterior_predictive[model.response_component.response.name] # type: ignore

    else:
        # PyMC model - use sample_posterior_predictive with updated data
        return _predict_pymc(
            model=model,
            idata=idata,
            data=data,
            kind=kind,
            origin_col=origin_col,
            dev_col=dev_col,
            calendar_col=calendar_col,
            exposure_col=exposure_col,
        )


def _predict_pymc(
    model: pm.Model,
    idata: az.InferenceData,
    data: pd.DataFrame,
    kind: Literal["mean", "response"] = "mean",
    origin_col: str = "origin",
    dev_col: str = "dev",
    calendar_col: str | None = None,
    exposure_col: str | None = None,
) -> xr.DataArray:
    """
    Generate posterior predictions for a PyMC model.

    For out-of-sample predictions, this function computes the linear predictor
    manually from the posterior samples and then generates predictions.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model.
    idata : az.InferenceData
        The inference data from model fitting.
    data : pd.DataFrame
        New data for predictions.
    kind : {"mean", "response"}, optional
        Type of prediction.
    origin_col : str, optional
        Name of origin column.
    dev_col : str, optional
        Name of development column.
    calendar_col : str, optional
        Name of calendar column.
    exposure_col : str, optional
        Name of exposure column.

    Returns
    -------
    xr.DataArray
        DataArray with posterior predictions.
    """
    # Get the original coords to map new data to indices
    origin_levels = list(model.coords["origin"]) # type: ignore
    dev_levels = list(model.coords["dev"]) # type: ignore

    # Encode new data using the same levels
    def encode_column(values: pd.Series, levels: list) -> np.ndarray:
        """Encode values to integer indices based on known levels."""
        level_to_idx = {level: idx for idx, level in enumerate(levels)}
        codes = np.array([level_to_idx.get(v, -1) for v in values])
        return codes

    origin_codes = encode_column(data[origin_col], origin_levels)
    dev_codes = encode_column(data[dev_col], dev_levels)

    # Check for unknown levels
    if (origin_codes == -1).any():
        unknown = data[origin_col][origin_codes == -1].unique()
        raise ValueError(f"Unknown origin levels in prediction data: {unknown}")
    if (dev_codes == -1).any():
        unknown = data[dev_col][dev_codes == -1].unique()
        raise ValueError(f"Unknown dev levels in prediction data: {unknown}")

    # Extract posterior samples
    posterior = idata.posterior # type: ignore

    # Get parameter arrays - stack chains and draws
    alpha_origin = posterior["alpha_origin"].values  # shape: (chains, draws, n_origin)
    alpha_dev = posterior["alpha_dev"].values  # shape: (chains, draws, n_dev)

    n_chains, n_draws = alpha_origin.shape[:2]
    n_obs = len(data)

    # Compute linear predictor for each posterior sample
    # mu[chain, draw, obs] = intercept + alpha_origin[origin_idx] + alpha_dev[dev_idx]
    mu = np.zeros((n_chains, n_draws, n_obs))

    # Add intercept if present
    if "intercept" in posterior:
        intercept = posterior["intercept"].values  # shape: (chains, draws)
        mu += intercept[:, :, np.newaxis]

    # Add origin effects
    for i, idx in enumerate(origin_codes):
        mu[:, :, i] += alpha_origin[:, :, idx]

    # Add development effects
    for i, idx in enumerate(dev_codes):
        mu[:, :, i] += alpha_dev[:, :, idx]

    # Add calendar effects if present
    if calendar_col is not None and "alpha_calendar" in posterior:
        calendar_levels = list(model.coords["calendar"]) # type: ignore
        calendar_codes = encode_column(data[calendar_col], calendar_levels)
        if (calendar_codes == -1).any():
            unknown = data[calendar_col][calendar_codes == -1].unique()
            raise ValueError(f"Unknown calendar levels in prediction data: {unknown}")

        alpha_calendar = posterior["alpha_calendar"].values
        for i, idx in enumerate(calendar_codes):
            mu[:, :, i] += alpha_calendar[:, :, idx]

    # Add log exposure offset if present
    if exposure_col is not None:
        log_exposure = np.log(np.asarray(data[exposure_col].values, dtype=np.float64))
        mu += log_exposure[np.newaxis, np.newaxis, :]

    # Apply inverse link (exp for log link)
    mu_exp = np.exp(mu)

    # Create xarray DataArray
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_draws),
        "obs": np.arange(n_obs),
    }

    if kind == "mean":
        # Return the expected value (mu after inverse link)
        mean_pred = mu_exp.mean(axis=(0, 1))
        return xr.DataArray(mean_pred, dims=["obs"], coords={"obs": coords["obs"]})
    else:
        # Return full posterior samples of predictions
        return xr.DataArray(mu_exp, dims=["chain", "draw", "obs"], coords=coords)


def posterior_predictive_check(
    model: bmb.Model,
    idata: az.InferenceData,
    n_samples: int = 500,
    random_seed: int | None = None,
) -> az.InferenceData:
    """
    Generate posterior predictive samples for model checking.

    Parameters
    ----------
    model : bmb.Model
        The fitted Bambi model.
    idata : az.InferenceData
        The inference data from model fitting.
    n_samples : int, optional
        Number of posterior predictive samples. Default is 500.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    az.InferenceData
        InferenceData with posterior_predictive group added.
    """
    model.predict(idata, kind="response", inplace=True)
    return idata


def compute_waic(idata: az.InferenceData) -> az.ELPDData:
    """
    Compute WAIC (Widely Applicable Information Criterion) for model comparison.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with log_likelihood group.

    Returns
    -------
    az.ELPDData
        WAIC computation results.
    """
    return az.waic(idata)


def compute_loo(idata: az.InferenceData) -> az.ELPDData:
    """
    Compute LOO-CV (Leave-One-Out Cross-Validation) for model comparison.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with log_likelihood group.

    Returns
    -------
    az.ELPDData
        LOO-CV computation results.
    """
    return az.loo(idata)


def build_csr_model(
    data: pd.DataFrame,
    logprem_col: str = "logprem",
    logloss_col: str = "logloss",
    origin_col: str = "origin",
    dev_col: str = "dev",
    priors: dict[str, Any] | None = None,
) -> pm.Model:
    """
    Build a PyMC model for the Changing Settlement Rate (CSR) method.

    This implements the CSR stochastic loss reserving method from Glenn Meyers'
    "Stochastic Loss Reserving Using Bayesian MCMC Models" (2015). The CSR model
    allows for changing development patterns over time, where newer accident years
    may settle at different rates than older ones.

    Model Structure
    ---------------
    The mean structure is:

        E[log(loss)] = logprem + logelr + alpha[origin] + beta[dev] * speedup[origin]

    where:
    - logprem is log premium (offset)
    - logelr is the log expected loss ratio
    - alpha[origin] are origin year effects (first constrained to 0)
    - beta[dev] are development effects (last constrained to 0)
    - speedup[origin] is a cumulative factor: speedup[1]=1, speedup[i]=speedup[i-1]*(1-gamma)

    The variance structure allows for heteroscedasticity across development periods,
    with variance typically decreasing as claims mature.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame with columns for log premium, log loss,
        origin period, and development period.
    logprem_col : str, optional
        Name of the log premium column. Default is "logprem".
    logloss_col : str, optional
        Name of the log loss column. Default is "logloss".
    origin_col : str, optional
        Name of the origin period column. Default is "origin".
    dev_col : str, optional
        Name of the development period column. Default is "dev".
    priors : dict, optional
        Custom prior specifications. Keys can include:
        - "alpha": dict with "sigma" for origin effects prior
        - "beta": dict with "sigma" for development effects prior
        - "logelr": dict with "mu" and "sigma" for log ELR prior
        - "gamma": dict with "mu" and "sigma" for speedup parameter prior
        - "a_ig": dict with "alpha" and "beta" for inverse gamma prior on variance

    Returns
    -------
    pm.Model
        A PyMC model object ready for sampling.

    References
    ----------
    Meyers, G. (2015). Stochastic Loss Reserving Using Bayesian MCMC Models.
    CAS Monograph Series Number 1.

    Examples
    --------
    >>> import pandas as pd
    >>> from bayesianchainladder.models import build_csr_model
    >>> data = pd.DataFrame({
    ...     "logprem": [10.0, 10.0, 10.0, 10.1, 10.1, 10.2],
    ...     "logloss": [8.0, 8.5, 8.8, 8.1, 8.6, 8.2],
    ...     "origin": [1, 1, 1, 2, 2, 3],
    ...     "dev": [1, 2, 3, 1, 2, 1],
    ... })
    >>> model = build_csr_model(data)
    """
    priors = priors or {}

    # Get data arrays
    logloss = data[logloss_col].values.astype(np.float64)
    logprem = data[logprem_col].values.astype(np.float64)
    n_obs = len(logloss)

    # Encode categorical variables (1-indexed to match Stan)
    origin_codes, origin_levels = pd.factorize(data[origin_col], sort=True)
    dev_codes, dev_levels = pd.factorize(data[dev_col], sort=True)

    n_origin = len(origin_levels)
    n_dev = len(dev_levels)

    coords = {
        "origin": origin_levels,
        "dev": dev_levels,
        "obs": np.arange(n_obs),
        "origin_raw": origin_levels[1:],  # For r_alpha (n_origin - 1)
        "dev_raw": dev_levels[:-1],  # For r_beta (n_dev - 1)
    }

    # Prior specifications
    alpha_sigma = priors.get("alpha", {}).get("sigma", 3.162)
    beta_sigma = priors.get("beta", {}).get("sigma", 3.162)
    logelr_mu = priors.get("logelr", {}).get("mu", -0.4)
    logelr_sigma = priors.get("logelr", {}).get("sigma", 3.162)
    gamma_mu = priors.get("gamma", {}).get("mu", 0.0)
    gamma_sigma = priors.get("gamma", {}).get("sigma", 0.05)
    a_ig_alpha = priors.get("a_ig", {}).get("alpha", 1.0)
    a_ig_beta = priors.get("a_ig", {}).get("beta", 1.0)

    with pm.Model(coords=coords) as model:
        # Data containers
        origin_idx = pm.Data("origin_idx", origin_codes, dims="obs")
        dev_idx = pm.Data("dev_idx", dev_codes, dims="obs")
        logprem_data = pm.Data("logprem", logprem, dims="obs")

        # ===== Parameters =====

        # Raw origin effects (n_origin - 1), first is constrained to 0
        r_alpha = pm.Normal("r_alpha", mu=0, sigma=alpha_sigma, dims="origin_raw")

        # Raw development effects (n_dev - 1), last is constrained to 0
        r_beta = pm.Normal("r_beta", mu=0, sigma=beta_sigma, dims="dev_raw")

        # Log expected loss ratio (constrained to [-4, 4] in Stan)
        # Using a normal with moderate sigma, or could use pm.Truncated
        logelr = pm.Normal("logelr", mu=logelr_mu, sigma=logelr_sigma)

        # Speedup parameter gamma
        gamma = pm.Normal("gamma", mu=gamma_mu, sigma=gamma_sigma)

        # Inverse gamma parameters for variance (one per development period)
        a_ig = pm.InverseGamma("a_ig", alpha=a_ig_alpha, beta=a_ig_beta, dims="dev")

        # ===== Transformed Parameters =====

        # alpha: first is 0, rest are r_alpha
        # alpha[0] = 0, alpha[1:] = r_alpha
        alpha = pt.concatenate([pt.zeros(1), r_alpha])
        alpha = pm.Deterministic("alpha", alpha, dims="origin")

        # beta: last is 0, rest are r_beta
        # beta[:-1] = r_beta, beta[-1] = 0
        beta = pt.concatenate([r_beta, pt.zeros(1)])
        beta = pm.Deterministic("beta", beta, dims="dev")

        # speedup: speedup[0] = 1, speedup[i] = speedup[i-1] * (1 - gamma)
        # This is a geometric sequence: speedup[i] = (1 - gamma)^i
        speedup_values = pt.power(1 - gamma, pt.arange(n_origin))
        speedup = pm.Deterministic("speedup", speedup_values, dims="origin")

        # Variance structure from Stan:
        # sig2[n_dev] = gamma_cdf(1/a_ig[n_dev], 1, 1)
        # sig2[n_dev-i] = sig2[n_dev+1-i] + gamma_cdf(1/a_ig[i], 1, 1)
        # This creates decreasing variance as development progresses

        # In PyMC, we use the Gamma distribution CDF
        # gamma_cdf(x, alpha=1, beta=1) = 1 - exp(-x) for alpha=beta=1
        # Note: Stan's gamma_cdf uses shape-rate parameterization

        # Compute cumulative variance from the last dev period backwards
        # First compute the individual contributions
        sig2_contrib = 1.0 - pt.exp(-1.0 / a_ig)  # gamma_cdf(1/a_ig, 1, 1)

        # Cumulative sum in reverse (from last dev to first)
        sig2_reversed = pt.cumsum(sig2_contrib[::-1])
        sig2 = sig2_reversed[::-1]
        sig = pm.Deterministic("sig", pt.sqrt(sig2), dims="dev")

        # ===== Mean Structure =====
        # mu[i] = logprem[i] + logelr + alpha[origin[i]] + beta[dev[i]] * speedup[origin[i]]

        mu = (
            logprem_data
            + logelr
            + alpha[origin_idx]
            + beta[dev_idx] * speedup[origin_idx]
        )
        mu = pm.Deterministic("mu", mu, dims="obs")

        # ===== Likelihood =====
        # logloss ~ Normal(mu, sig[dev_lag])
        pm.Normal("logloss", mu=mu, sigma=sig[dev_idx], observed=logloss, dims="obs")

    return model


def extract_parameter_summary(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    filter_vars: str | None = None,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """
    Extract summary statistics for model parameters.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with posterior samples.
    var_names : list[str], optional
        Parameter names to include. If None, includes all.
    hdi_prob : float, optional
        Probability mass for HDI. Default is 0.94.

    Returns
    -------
    pd.DataFrame
        Summary statistics for parameters.
    """
    return az.summary(idata, var_names=var_names, filter_vars=filter_vars, hdi_prob=hdi_prob) # type: ignore


def sample_prior_predictive(
    model: bmb.Model,
    draws: int = 500,
    random_seed: int | None = None,
) -> az.InferenceData:
    """
    Sample from the prior predictive distribution.

    Prior predictive checks are essential for validating that priors produce
    reasonable predictions before fitting the model to data. This is especially
    important in loss reserving where domain knowledge about loss magnitudes,
    development patterns, and reserve ranges should inform prior selection.

    Parameters
    ----------
    model : bmb.Model
        A Bambi model (unfitted or fitted).
    draws : int, optional
        Number of prior predictive samples per chain. Default is 500.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    az.InferenceData
        InferenceData object with prior and prior_predictive groups containing:
        - prior: Samples from prior distributions of all parameters
        - prior_predictive: Samples of predicted responses under the prior

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder import BayesianChainLadderGLM
    >>> from bayesianchainladder.models import sample_prior_predictive
    >>>
    >>> # Create model but don't fit
    >>> model = BayesianChainLadderGLM(draws=1000)
    >>> model._build_model(triangle)  # Build without fitting
    >>>
    >>> # Sample from prior predictive
    >>> prior_idata = sample_prior_predictive(model.model_, draws=500)
    >>>
    >>> # Examine prior predictions
    >>> import arviz as az
    >>> az.plot_ppc(prior_idata, group="prior")

    Notes
    -----
    For loss reserving, prior predictive checks help verify that:
    1. Predicted incremental losses are within plausible ranges
    2. Development patterns are reasonable (most development early)
    3. Reserve estimates are not absurdly large or negative
    4. Loss ratios (if exposure available) are within industry norms
    """
    # Ensure the model is built before sampling
    # Bambi requires model.build() to be called before prior_predictive()
    if not model.backend:
        model.build()

    # Use Bambi's prior predictive sampling capability
    idata = model.prior_predictive(draws=draws, random_seed=random_seed)

    return idata


def compute_prior_predictive_summary(
    prior_idata: az.InferenceData,
    response_name: str = "incremental",
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics from prior predictive samples.

    Parameters
    ----------
    prior_idata : az.InferenceData
        InferenceData with prior_predictive group.
    response_name : str, optional
        Name of the response variable. Default is "incremental".
    quantiles : list[float], optional
        Quantiles to compute. Default is [0.025, 0.25, 0.5, 0.75, 0.975].

    Returns
    -------
    pd.DataFrame
        Summary statistics including mean, std, and quantiles.
    """
    if quantiles is None:
        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    if "prior_predictive" not in prior_idata.groups():
        raise ValueError("InferenceData must contain prior_predictive group")

    # Get prior predictive samples
    pp = prior_idata.prior_predictive[response_name]

    # Stack chains and draws
    pp_flat = pp.stack(sample=["chain", "draw"])

    # Compute summary statistics per observation
    summary_data = {
        "mean": pp_flat.mean(dim="sample").values,
        "std": pp_flat.std(dim="sample").values,
    }

    for q in quantiles:
        q_label = f"{q*100:.1f}%"
        summary_data[q_label] = pp_flat.quantile(q, dim="sample").values

    return pd.DataFrame(summary_data)
