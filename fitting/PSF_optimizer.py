import torch
from typing import Optional
from torchmin import minimize


def OptimizePSFModel(
    PSF_model,
    loss_fn,
    x_initial = None,
    include_params = None,
    *,
    max_iter:    int   = 300,
    n_attempts:  int   = 1,
    loss_thresh: float = 1e1,
    verbose:     bool  = False,
    force_bfgs:  bool  = False,
    perturb_between_attempts: bool = False,
    perturb_sigma: float = 1e-2,
    perturb_decay: float = 0.7,
    rng_seed: Optional[int] = None,
):
    """
    Optimize PSF model parameters:
      1. L-BFGS first (or BFGS directly if forced)
      2. Retry with BFGS when it stops too early with high loss
      3. Returns optimized parameter vector and success flag
    """
    all_params_init = PSF_model.get_param_names()
    optimizable_params_init = PSF_model.get_optimizable_param_names()
    
    if include_params is None:
        if verbose:
            print("No include_params specified. Optimizing all optimizable parameters in PSF_model.inputs_manager.")
        include_params = optimizable_params_init

    if   isinstance(include_params, str): include_params = [include_params]
    elif isinstance(include_params, set): include_params = list(include_params)
    
    if len(include_params) == 0:
        raise ValueError("No optimizable parameters specified for optimization. Please check include_params list.")

    # Temporarily disable everything and enable only requested parameters.
    PSF_model.inputs_manager.set_optimizable(all_params_init, False)
    PSF_model.inputs_manager.set_optimizable(include_params, True)

    try:
        n_attempts = max(1, int(n_attempts))
        x_start = PSF_model.inputs_manager.stack().clone() if x_initial is None else x_initial

        def _run(method, tolerance, x0):
            return minimize(
                loss_fn,
                x0,
                max_iter = max_iter,
                tol = tolerance,
                method = method,
                disp = 2 if verbose else 0,
            )

        # -- Runs up to n_attempts times
        # -- Each new attempt starts from the best solution found so far.
        # -- Optional random perturbation between attempts can help escape local minima.
        # -- BFGS fallback for "stopped too early + high loss" is preserved per attempt
        # -- Tracks best result across attempts and returns it
        # -- After choosing the best result, it updates PSF_model.inputs_manager with result.x

        rng = None
        if perturb_between_attempts and rng_seed is not None:
            rng = torch.Generator(device=x_start.device)
            rng.manual_seed(rng_seed)

        best_result = None
        success = False

        for i_attempt in range(n_attempts):
            if verbose and n_attempts > 1:
                print(f"Attempt {i_attempt+1}/{n_attempts}")

            _ = PSF_model.inputs_manager.unstack(x_start, include_all=True, update=True)
            result = _run("bfgs", 1e-4, x_start) if force_bfgs else _run("l-bfgs", 1e-4, x_start)

            stopped_too_early = result["nit"] < max_iter * 0.3
            loss_too_high = result["fun"] > loss_thresh
            
            if stopped_too_early and loss_too_high and not force_bfgs:
                if verbose:
                    print("Warning: minimization likely stopped too early with high loss. Trying BFGS...")
                _ = PSF_model.inputs_manager.unstack(x_start, include_all=True, update=True)
                result = _run("bfgs", 1e-5, x_start)

            if best_result is None or result["fun"] < best_result["fun"]:
                best_result = result

            # Always perform all attempts and restart from the best known point.
            x_start = best_result.x.clone()

            if perturb_between_attempts and i_attempt < n_attempts - 1:
                sigma_k = perturb_sigma * (perturb_decay ** i_attempt)
                if sigma_k > 0:
                    noise = torch.randn(x_start.shape, device=x_start.device, dtype=x_start.dtype, generator=rng)
                    x_start = x_start + noise * sigma_k
                    if verbose:
                        print(f"Applied restart perturbation: sigma={sigma_k:.3e}")

        if verbose:
            print("-" * 50)

        if best_result is None:
            raise RuntimeError("Optimization did not run any attempts.")

        result  = best_result
        success = result["fun"] < loss_thresh
        if not success:
            print("Warning: Minimization did not converge.")

        _ = PSF_model.inputs_manager.unstack(result.x, include_all=True, update=True)

        return result.x, success

    finally:
        # Restore original optimizable/fixed states.
        PSF_model.inputs_manager.set_optimizable(all_params_init, False)
        PSF_model.inputs_manager.set_optimizable(optimizable_params_init, True)