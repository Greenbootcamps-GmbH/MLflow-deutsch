
# Ridge Param Grid
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
}

elasticnet_param_grid = {
    'alpha': [0.01, 0.1, 1.0],
    'l1_ratio': [0.2, 0.5, 0.8],
    'fit_intercept': [True, False],
}
