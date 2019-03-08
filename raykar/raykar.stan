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
 *   outcome: 0: genuine; 1: deceptive
 *   heuristics act as "annotators"
 *   
 */
data {
  int<lower=0> N;               // # items (reviews)
  int<lower=0> R;               // # raters (heuristics)
  int<lower=0> D;               // # of predictors for each item
  vector[D] x[N];               // predictors for item n in 1:N
  int<lower=0,upper=1> y[N,R];  // 0 if genuine, 1 if deceptive
}

parameters {
  int<lower=0,upper=1> z[N];          // true status of review n
  vector[D] w;                        // logistic regression coefficients
  vector<lower=0,upper=1> alpha[R];   // sensitivity of r in 1;R
  vector<lower=0,upper=1> beta[R];    // specificity of r in 1:R
}

model {
  // priors
  w ~ normal(0,5); 
  alpha ~ beta(2,2);
  beta ~ beta(2,2); 

  // classification model
  for (i in 1:N)
    z[i] ~ bernoulli(inv_logit(w' * x[i]));


  // "two coin" model
  for (i in 1:N) {
    for (r in 1:R) {
      if (z[i] == 1)
        y[i,r] ~ bernoulli(alpha[r]);  // sensitivity if z[n] == 1
      else
        y[i,r] ~ bernoulli(1 - beta[r]);   // specificition if z[n] == 0
    }
  }  
}
  

