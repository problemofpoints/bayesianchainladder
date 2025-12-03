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
    )

    return model


def _get_family(family: str, link: str | None = None) -> str:
    """Get Bambi family specification.

    Note: In Bambi 0.13+, we just return the family name as a string.
    Bambi will use its default link function for each family.
    Custom links require creating a full Family object with Likelihood,
    which is complex and rarely needed for standard actuarial models.
    """
    # Map family names to Bambi family names
    family_map = {
        "negativebinomial": "negativebinomial",
        "negative_binomial": "negativebinomial",
        "negbinom": "negativebinomial",
        "poisson": "poisson",
        "gamma": "gamma",
        "gaussian": "gaussian",
        "normal": "gaussian",
    }

    family_lower = family.lower()
    if family_lower not in family_map:
        raise ValueError(
            f"Unknown family '{family}'. Supported families: {list(family_map.keys())}"
        )

    bambi_family = family_map[family_lower]

    # For now, just return the family name - Bambi uses sensible defaults
    # gaussian uses identity link, others use log link
    if link is not None and link != _get_default_link(bambi_family):
        import warnings
        warnings.warn(
            f"Custom link '{link}' specified but Bambi will use its default link for '{bambi_family}'. "
            "Custom links require advanced Family configuration.",
            UserWarning,
        )

    return bambi_family


def _get_default_link(family: str) -> str:
    """Get the default link function for a family."""
    default_links = {
        "negativebinomial": "log",
        "poisson": "log",
        "gamma": "inverse",
        "gaussian": "identity",
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
