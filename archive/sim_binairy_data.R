library(dplyr)
library(rstan)

# Before we observe we have to deel with:
I <- 70 #items
J <- 10 # observations 
K <- 2 # number of classes 
N <- 100 # number of annotation, not every annotator has to annotate all items or some can annotate multiple times


# 1) true class  - c
# which is estiamted from class prevalence pi
# pi can itself be associated with a model e.g. via inverse logit. 

Upsilon <- mvtnorm::rmvnorm(n = I, mean = c(1, 1, 1), sigma = diag(3))
B <- matrix(c(1, 2, -1), nrow = 3)

inv_logit <- function(x) exp(x) / (1 + exp(x))
pi <- inv_logit(Upsilon%*%B)
c <- rbinom(I, 1, prob = pi)

# 2) rater ability - beta
# which can itself have a model
# lets use specificity and sensitivity
spec <- .85
sens <- .75

# Beta <- array(NA, dim = c(J, K, K))
# for(j in 1:J){
#   for(k in 1:K){
#     Beta[j, k, ] <- rep(0.2 / (K - 1), K)
#     Beta[j , k, k] <- 0.8
#   }
# }


# sens and spec before probs
# # than both together cause an interaction that gets us observed y's
# # where y ~ bernoulli(beta, z)
# probs.y1 <- pi %*% spec + (1-pi) %*% (1-sens)
# # probs.y0 <- (1-pi) %*% sens + pi %*% (1-spec)  
# y <- NULL
# for(j in 1:J) y <- c(y, rbinom(I, 1, probs.y1))  

# sens and spec after probs
y <- NULL
for(j in 1:J) {
  for(i in 1:I){
    if(c[i] == 0) {
      y[i + (I*(j-1))] <- rbinom(1, 1, prob = (1-sens))
    } else {
      y[i + (I*(j-1))] <- rbinom(1, 1, prob = spec)
    }
  }
}

sim.data <- cbind(rep(1:I, J), rep(1:J, each = I), y)
colnames(sim.data) <- c("item", "annotator", "rating")  


DS_em <- function(J, K, N, I, ii, jj, y){
  ##### EM ALGORITHM #####
  
  ### INITIALIZATION
  theta_hat <- array(NA,c(J,K,K));
  for (j in 1:J)
    for (k in 1:K)
      for (k2 in 1:K)
        theta_hat[j,k,k2] <- ifelse(k==k2, 0.7, 0.3/K);
      
      pi_hat <- array(1/K,K);
      
      ### EM ITERATIONS
      epoch <- 1;
      min_relative_diff <- 1E-8;
      last_log_posterior = - Inf;
      E_z <- array(1/K, c(I,K));
      MAX_EPOCHS <- 100;
      for (epoch in 1:MAX_EPOCHS) {
        ### E step 
        for (i in 1:I)
          E_z[i,] <- pi_hat;
        for (n in 1:N)
          for (k in 1:K)
            E_z[ii[n],k] <- E_z[ii[n],k] * theta_hat[jj[n],k,y[n]];
          for (i in 1:I)
            E_z[i,] <- E_z[i,] / sum(E_z[i,]);
          
          ### M step
          beta <- 0.01; 
          pi_hat <- rep(beta,K);          # add beta smoothing on pi_hat
          for (i in 1:I)
            pi_hat <- pi_hat + E_z[i,];
          pi_hat <- pi_hat / sum(pi_hat);
          
          alpha <- 0.01;
          count <- array(alpha,c(J,K,K)); # add alpha smoothing for theta_hat
          for (n in 1:N)
            for (k in 1:K)
              count[jj[n],k,y[n]] <- count[jj[n],k,y[n]] + E_z[ii[n],k];
            for (j in 1:J)
              for (k in 1:K)
                theta_hat[j,k,] <- count[j,k,] / sum(count[j,k,]);
              
              p <- array(0,c(I,K));
              for (i in 1:I)
                p[i,] <- pi_hat;
              for (n in 1:N)
                for (k in 1:K)
                  p[ii[n],k] <- p[ii[n],k] * theta_hat[jj[n],k,y[n]];
                log_posterior <- 0.0;
                for (i in 1:I)
                  log_posterior <- log_posterior + log(sum(p[i,]));
                if (epoch == 1)
                  print(paste("epoch=",epoch," log posterior=", log_posterior));
                if (epoch > 1) {
                  diff <- log_posterior - last_log_posterior;
                  relative_diff <- abs(diff / last_log_posterior);
                  print(paste("epoch=",epoch,
                              " log posterior=", log_posterior,
                              " relative_diff=",relative_diff));
                  if (relative_diff < min_relative_diff) {
                    print("FINISHED.");
                    break;
                  }
                }
                last_log_posterior <- log_posterior;
      }
      
      
      sum(apply((E_z>.99),FUN = purrr::has_element, MARGIN = 1, .y=TRUE)) / I
      
      
      
      
      # VOTED PREVALENCE AS A SANITY CHECK; compare to estimates of pi
      voted_prevalence <- rep(0,K);
      for (k in 1:K)
        voted_prevalence[k] <- sum(y == k);
      voted_prevalence <- voted_prevalence / sum(voted_prevalence);
      print(paste("voted prevalence=",voted_prevalence));
      
      pi_out <- array(0,dim=c(K,2),dimnames=list(NULL,c("category","prob")));
      pos <- 1;
      for (k in 1:K) {
        pi_out[pos,] <- c(k,pi_hat[k]);
        pos <- pos + 1;
      }
      # write.table(pi_out,sep='\t',row.names=FALSE,file="pi_hat.tsv",quote=FALSE);
      
      theta_out <- array(0,
                         dim=c(J*K*K,4),
                         dimnames=list(NULL,c("annotator","reference",
                                              "response","prob")));
      pos <- 1;
      for (j in 1:J) {
        for (ref in 1:K) {
          for (resp in 1:K) {
            theta_out[pos,] <- c(j,ref,resp,theta_hat[j,ref,resp]);
            pos <- pos + 1;
          }
        }
      }
      # write.table(theta_out,
      #             sep='\t',
      #             row.names=FALSE,
      #             file="theta_hat.tsv",quote=FALSE);
      
      z_out <- array(0,dim=c(I*K,3),
                     dimnames=list(NULL,c("item","category","prob")));
      pos <- 1;
      for (i in 1:I) {
        for (k in 1:K) {
          z_out[pos,] = c(i,k,E_z[i,k]);
          pos <- pos + 1;
        }
      }
      # write.table(z_out,sep='\t',row.names=FALSE,file="z_hat.tsv",quote=FALSE);
      
      output <- list(pi_out = pi_out,
                     beta_out = theta_out,
                     c_out = z_out)
}

fit_DS_em <- DS_em(J  = J,
                   K  = K,
                   N  = N,
                   I  = I,
                   ii = sim.data[, 1],
                   jj = sim.data[, 2],
                   y  = (sim.data[, 3] + 1))

fit_DS_em$pi_out[, 1] <- fit_DS_em$pi_out[, 1] - 1
fit_DS_em$beta_out[, 3] <- fit_DS_em$beta_out[, 3] - 1
fit_DS_em$c_out[, 2] <- fit_DS_em$c_out[, 2] - 1

# fit_DS_em
fit_DS_em$pi_out
sum(c)/length(c)

c.matrix <- round(t(matrix(fit_DS_em$c_out[,3], ncol = I, nrow = K)),2)
c.matrix
yhat <- NULL
for(i in 1:I) if(round(c.matrix[i, 1]) == 1) yhat[i] <- 0 else yhat[i] <- 1
sum(yhat == y) / length(y)
for(mat in 1:J) print( round(t(matrix(fit_DS_em$beta_out[(((mat-1)*(K^2))+1):(mat*(K^2)) ,4],ncol=K)), 2))


# head(sim.data)
# DS_bayes <- stan_model("Comparing Bayesian Models of Annotation/DS.stan")
# saveRDS(DS_bayes, file = "ds_stan.rds")
DS_bayes <- readRDS("ds_stan.rds")
pi_init <- table(sim.data[, 3]) / sum(table(sim.data[, 3]))
beta_init <- array(NA, dim = c(J, K, K))
for(j in 1:J){
  for(k in 1:K){
    beta_init[j, k, ] <- rep(0.1 / (K - 1), K)
    beta_init[j , k, k] <- 0.9
  }
}


init_fun <- function(n) {
  list(pi = pi_init,
       beta = beta_init)
}

fit_DS_bayes <- sampling(object = DS_bayes, data = list(J  = J,
                                                        K  = K,
                                                        N  = N,
                                                        I  = I,
                                                        ii = sim.data[, 1],
                                                        jj = sim.data[, 2],
                                                        y  = (sim.data[, 3] + 1)),
                         seed = 13031990, chains = 4, init = init_fun)

fit_DS_bayes




## sim.data with predictors

# c.matrix
# yhat
log.odds <- log(c.matrix[, 2] / c.matrix[, 1])
fit.glm <- glm(yhat ~ -1 + Upsilon, family = binomial())
fit.lm <- lm(log.odds ~ -1 + Upsilon)
sum((predict(fit.glm, type = "response") - pi)^2)
sum(((1 / (1 +  exp(-predict(fit.lm)))) - pi)^2)


