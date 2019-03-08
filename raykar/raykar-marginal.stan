/**
 * Binary Raykar et al. notation translation
 *    Raykar et al  =  this model
 *    D_i == (x[i], y[i,])
 *    y^r_n == y[r,n]
 *    y_i == z[i]
 *
 * Raykar et al. specify priors:
 *    alpha[j] ~ beta(a1[j], a2[j]);
 *    beta[j] ~ beta(b1[j], b2[j]);
 *    w ~ multi_normal(rep_vector(0,D), inverse(Gamma));
 * but fail to mention values of a1, a2, b1, b2, Gamma
 *
 * applied by Poesio et al. to reviews on Amazon
 *   items are reviews
 *   heuristics act as "annotators"
 *
 */
data {
  int<lower=0> N;               // # items (reviews)
  int<lower=0> R;               // # raters (heuristics)
  int<lower=0> D;               // # of predictors for each item
  matrix[N,D] x;                // [n,d] predictors d for item n
  int<lower=0,upper=1> y[N,R];  // 0 if genuine, 1 if deceptive
}
parameters {
  // constrain sensitivity and specificity to be non-adversarial
  vector[D] w;                          // logistic regression coeffs
  real w0;                              // intercept

  vector<lower=0.1,upper=1>[R] alpha;   // sensitivity
  vector<lower=0.1,upper=1>[R] beta;    // specificity
}
model {
  vector[N] logit_z_hat;
  vector[N] log_z_hat;   // log_z[n] = log Pr[z[n] == 1]
  vector[N] log1m_z_hat; // log_z[n] = log Pr[z[n] == 0]

  vector[R] log_alpha;
  vector[R] log1m_alpha;
  vector[R] log_beta;
  vector[R] log1m_beta;

  logit_z_hat <- w0 + x * w;
  for (n in 1:N) {
    log_z_hat[n] <- log_inv_logit(logit_z_hat[n]);
    log1m_z_hat[n] <- log1m_inv_logit(logit_z_hat[n]);
  }

  for (r in 1:R) {
    log_alpha[r] <- log(alpha[r]);
    log1m_alpha[r] <- log1m(alpha[r]);
    log_beta[r] <- log(beta[r]);
    log1m_beta[r] <- log1m(beta[r]);
  }

  // priors
  w ~ normal(0,2);
  w0 ~ normal(0,5);
  alpha ~ beta(10,1);
  beta ~ beta(10,1);

  for (n in 1:N) {
    real pos_sum;
    real neg_sum;
    pos_sum <- log_z_hat[n];
    neg_sum <- log1m_z_hat[n];
    for (r in 1:R) {
      if (y[n,r] == 1) {
        pos_sum <- pos_sum + log_alpha[r];
        neg_sum <- neg_sum + log1m_beta[r];
      } else {
        pos_sum <- pos_sum + log1m_alpha[r];
        neg_sum <- neg_sum + log_beta[r];
      }
    }
    increment_log_prob(log_sum_exp(pos_sum, neg_sum));
  }
}
generated quantities {
  vector[N] E_z;
  {
    vector[N] logit_z_hat;
    vector[N] log_z_hat;   // posterior mean: log Pr[z[n] == 1]
    vector[N] log1m_z_hat; //                 log Pr[z[n] == 0]

    vector[R] log_alpha;
    vector[R] log1m_alpha;
    vector[R] log_beta;
    vector[R] log1m_beta;

    logit_z_hat <- w0 + x * w;
    for (n in 1:N) {
      log_z_hat[n] <- log_inv_logit(logit_z_hat[n]);
      log1m_z_hat[n] <- log1m_inv_logit(logit_z_hat[n]);
    }

    for (r in 1:R) {
      log_alpha[r] <- log(alpha[r]);
      log1m_alpha[r] <- log1m(alpha[r]);
      log_beta[r] <- log(beta[r]);
      log1m_beta[r] <- log1m(beta[r]);
    }

    for (n in 1:N) {
      real pos_sum;
      real neg_sum;
      real maxpn;
      pos_sum <- log_z_hat[n];
      neg_sum <- log1m_z_hat[n];
      for (r in 1:R) {
        if (y[n,r] == 1) {
          pos_sum <- pos_sum + log_alpha[r];
          neg_sum <- neg_sum + log1m_beta[r];
        } else {
          pos_sum <- pos_sum + log1m_alpha[r];
          neg_sum <- neg_sum + log_beta[r];
        }
      }
      maxpn <- fmax(pos_sum, neg_sum);
      pos_sum <- pos_sum - maxpn;
      neg_sum <- neg_sum - maxpn;
      E_z[n] <- exp(pos_sum) / (exp(pos_sum) + exp(neg_sum));
    }
  }
}
