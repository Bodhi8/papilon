def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full Bayes-factor fit, takes a few minutes")
