DS_bayes <- readRDS("ds_stan.rds")
set.seed(1)
get.coverage.sim <- function(I, J, K, sens, spec, missing){

I <- I #items
J <- J # annotators 
K <- K # number of classes 
missing <- missing

gamma <- mvtnorm::rmvnorm(n = I, mean = c(1, 1, 1), sigma = diag(3))
B <- matrix(c(1, 2, -1), nrow = 3)

inv_logit <- function(x) exp(x) / (1 + exp(x))
pi <- inv_logit(gamma%*%B)
c <- rbinom(I, 1, prob = pi)

spec <- spec
sens <- sens

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

sim.data.missing <- sim.data[-sample(1:nrow(sim.data), nrow(sim.data) * missing), ]

N <- nrow(sim.data.missing)



pi_init <- table(sim.data.missing[, 3]) / sum(table(sim.data.missing[, 3]))
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
                                                        ii = sim.data.missing[, 1],
                                                        jj = sim.data.missing[, 2],
                                                        y  = (sim.data.missing[, 3] + 1)),
                         seed = 13031990, chains = 4, init = init_fun)

out <- summary(fit_DS_bayes)$summary[3:(K*K*J+2), c(4, 8)]
coverage.sens <- NULL
for(j in 1:J) {
if(out[seq(1, (K*K*J), by = (K*K)), ][j, 1] < sens & out[seq(1, (K*K*J), by = (K*K)), ][j, 2] > sens) {
  coverage.sens[j] <- TRUE
} else {
  coverage.sens[j] <- FALSE
}
}

coverage.spec <- NULL
for(j in 1:J) {
  if(out[seq((K*K), (K*K*J), by = (K*K)), ][j, 1] < spec & out[seq((K*K), (K*K*J), by = (K*K)), ][j, 2] > spec) {
    coverage.spec[j] <- TRUE
  } else {
    coverage.spec[j] <- FALSE
  }
}

return(paste("Coverage sensitivity:", sum(coverage.sens) / length(coverage.sens),
             "Coverage specificity:", sum(coverage.spec) / length(coverage.spec)))
  
  
}

set.seed(13031990)
n.sim <- 50
sim.out <- NULL
for(sim in 1:n.sim) sim.out[sim] <- get.coverage.sim(I = 100, J = 50, K = 2, sens = .7, spec = .9, missing = .7)



