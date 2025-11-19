import gpytorch

print(gpytorch.settings.fast_computations.covar_root_decomposition.on())
print(gpytorch.settings.fast_computations.log_prob.on())
print(gpytorch.settings.fast_computations.solves.on())
print(gpytorch.settings.fast_pred_var.on())
print(gpytorch.settings.fast_pred_samples.on())
print(gpytorch.settings.max_cholesky_size.value())
print(gpytorch.settings.lazily_evaluate_kernels.on())
print(gpytorch.settings.memory_efficient.on())
print(gpytorch.settings.use_toeplitz.on())