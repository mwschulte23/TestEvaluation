# TestEvaluation
A/B testing package for continuous and binary metrics

Currently contains:
* Continuous frequentist bootsrapped p-value, bootstrapped confidence interval, and quantiled treatment methods
* Binary frequentist p-value and confidence interval methods

Bootstrapping is an A/B testing method that is more robust to various data distributions compared to standard t-test

Plan to add:
* Visualization methods including p-value trends over time
* Pretesting functionality
* Bayesian A/B testing methods

Example:
```test_instance = ContinuousTestEval(control_data, test_data)
test_instance.continuous_pval(n = 10000)```

