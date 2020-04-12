data {
    // Order of AR process
    int<lower=0> P;
    // Numer of observations
    int<lower=0> N;
    // Observations
    vector[N] y;
}
parameters {
    // Additive constant
    real alpha;                       
    // Coefficients
    vector[P] beta;
    // Standard deviation of the noise
    real<lower=0> sigma;              
}
transformed parameters {
    // Consider mu as a transform of the data, alpha and beta
    vector[N] mu;

    // Initial values
    mu[1:P] = y[1:P];

    for (t in (P + 1):N) {
        mu[t] = alpha;
        for (p in 1:P) {
            mu[t] += beta[p] * y[t-p];
        }
    }
}
model {
    // Increment the log-posterior
    y ~ normal(mu, sigma);
}
generated quantities {
    // Generate posterior predictives
    vector[N] y_pred;

    // First P points are known
    y_pred[1:P] = y[1:P];

    // Posterior predictive
    y_pred[(P + 1):N] = to_vector(normal_rng(mu[(P + 1):N], sigma));
}
