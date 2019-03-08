data {
  int<lower=1> I; //number of items
  int<lower=2> K; //number of classes
  int<lower=1> J; //number of annotators
  int<lower=1> N; //number of annotations
  int<lower=1, upper=I> ii[N]; //the item the n-th annotation belongs to
  int<lower=1, upper=J> jj[N]; //the annotator which produced the n-th annotation
  int<lower=0> D;               // # of predictors for each item
  vector[D] x[N];               // predictors for item n in 1:N
  int y[N]; //the class of the n-th annotation
}

transformed data {
  vector[K] alpha = rep_vector(1, K); //class prevalence prior
  vector[K] eta[K]; //annotator abilities prior
  for (k in 1:K) {
    eta[k] = rep_vector(1, K); 
    eta[k, k] = 2 * K; //make priors stronger on diagonal
  }
}

parameters {
  simplex[K] pi;
  simplex[K] beta[J, K];
  vector[D] w;                    // logistic regression coefficients
  //vector<lower=0,upper=1> alpha[J];   // sensitivity of j in 1;J
  //vector<lower=0,upper=1> beta[J];    // specificity of j in 1:J
}

transformed parameters {
  vector[K] log_q_z[I];
  vector[K] log_pi;
  vector[K] log_beta[J, K];
  vector[K] z_probs[I];
  
  log_pi = log(pi);
  log_beta = log(beta);
  //log_alpha = log(alpha);
  
  for (i in 1:I) 
    log_q_z[i] = log_pi;
  
  for (n in 1:N) 
    for (h in 1:K)
      log_q_z[ii[n], h] = log_q_z[ii[n], h] + log_beta[jj[n], h, y[n]];
  
  for(i in 1:I)
    z_probs[i] = softmax(log_q_z[i]);
    
}


model {
  for(j in 1:J)
    for(k in 1:K)
      beta[j, k] ~ dirichlet(eta[k]);
  
  pi ~ inv_logit(w' * x[i]);
  
  for (i in 1:I){
    target += log_sum_exp(log_q_z[i]);
  }
}

generated quantities {
  vector[K] q_z[I]; //the true class distribution of each item
  
  for(i in 1:I)
    q_z[i] = softmax(log_q_z[i]);
}
