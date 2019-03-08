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
  int<lower=0> I;               // # items (reviews)
  int<lower=0> J;               // # raters (heuristics)
  int<lower=0> D;               // # of predictors for each item
  matrix[I,D] x;                // [n,d] predictors d for item n
  int<lower=0,upper=1> y[I,J];  // 0 if genuine, 1 if deceptive
}
parameters {
  // constrain sensitivity and specificity to be non-adversarial
  vector[D] w;                          // logistic regression coeffs
  real w0;                              // intercept

  vector<lower=0.1,upper=1>[J] alpha;   // sensitivity
  vector<lower=0.1,upper=1>[J] beta;    // specificity
}

model {
  vector[I] logit_z_hat;
  vector[I] log_z_hat;   // log_z[n] = log Pr[z[n] == 1]
  vector[I] log1m_z_hat; // log_z[n] = log Pr[z[n] == 0]

  vector[J] log_alpha;
  vector[J] log1m_alpha;
  vector[J] log_beta;
  vector[J] log1m_beta;

  logit_z_hat = w0 + x * w;
  for (i in 1:I) {
    log_z_hat[i] = log_inv_logit(logit_z_hat[i]);
    log1m_z_hat[i] = log1m_inv_logit(logit_z_hat[i]);
  }

  for (j in 1:J) {
    log_alpha[j] = log(alpha[j]);
    log1m_alpha[j] = log1m(alpha[j]);
    log_beta[j] = log(beta[j]);
    log1m_beta[j] = log1m(beta[j]);
  }

  // priors
  w ~ normal(0,2);
  w0 ~ normal(0,5);
  alpha ~ beta(10,1);
  beta ~ beta(10,1);

  for (i in 1:I) {
    real pos_sum;
    real neg_sum;
    pos_sum = log_z_hat[i];
    neg_sum = log1m_z_hat[i];
    for (j in 1:J) {
      if (y[i,j] == 1) {
        pos_sum = pos_sum + log_alpha[j];
        neg_sum = neg_sum + log1m_beta[j];
      } else {
        pos_sum = pos_sum + log1m_alpha[j];
        neg_sum = neg_sum + log_beta[j];
      }
    }
    target += log_sum_exp(pos_sum, neg_sum);
  }
}
generated quantities {
  vector[I] E_z;
  {
    vector[I] logit_z_hat;
    vector[I] log_z_hat;   // posterior mean: log Pr[z[n] == 1]
    vector[I] log1m_z_hat; //                 log Pr[z[n] == 0]

    vector[J] log_alpha;
    vector[J] log1m_alpha;
    vector[J] log_beta;
    vector[J] log1m_beta;

    logit_z_hat = w0 + x * w;
    for (i in 1:I) {
      log_z_hat[i] = log_inv_logit(logit_z_hat[i]);
      log1m_z_hat[i] = log1m_inv_logit(logit_z_hat[i]);
    }

    for (j in 1:J) {
      log_alpha[j] = log(alpha[j]);
      log1m_alpha[j] = log1m(alpha[j]);
      log_beta[j] = log(beta[j]);
      log1m_beta[j] = log1m(beta[j]);
    }

    for (i in 1:I) {
      real pos_sum;
      real neg_sum;
      real maxpn;
      pos_sum = log_z_hat[i];
      neg_sum = log1m_z_hat[i];
      for (j in 1:J) {
        if (y[i,j] == 1) {
          pos_sum = pos_sum + log_alpha[j];
          neg_sum = neg_sum + log1m_beta[j];
        } else {
          pos_sum = pos_sum + log1m_alpha[j];
          neg_sum = neg_sum + log_beta[j];
        }
      }
      maxpn = fmax(pos_sum, neg_sum);
      pos_sum = pos_sum - maxpn;
      neg_sum = neg_sum - maxpn;
      E_z[i] = exp(pos_sum) / (exp(pos_sum) + exp(neg_sum));
    }
  }
}
