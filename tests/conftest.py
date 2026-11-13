def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full Bayes-factor fit, takes a few minutes")
    config.addinivalue_line("markers", 'jax: needs the optional JAX backend (pip install "papilon[jax]")')
