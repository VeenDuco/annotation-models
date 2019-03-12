library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# model_predictors <- stan_model("raykar/raykar_marginal_long.stan")
# saveRDS(model_predictors, "raykar/raykar_marginal_long.RDS")
model_predictors <- readRDS("raykar/raykar_marginal_long.RDS")
inv_logit <- function(u) 1 / (1 + exp(-u))

# SSE_twostep <- NULL
# SSE_direct <- NULL
# nsim <- 10
set.seed(13031990)

data.simulation <- function(I, K, J, D, missing){
  succes <- FALSE
  while(!succes){
    x <- matrix(rnorm(I * D), I, D)
    w <- rnorm(D)
    w0 <- -2
    
    alpha <- runif(J, 0.65, 0.95)
    beta <- runif(J, 0.65, 0.95)
    
    z <- rep(0, I);
    for (i in 1:I)
      z[i] <- rbinom(1, 1, inv_logit(w0 + (x[i, ] %*% w)))
    
    y <- matrix(0, I, J);
    for (i in 1:I)
      for (j in 1:J)
        y[i, j] <- rbinom(1, 1, ifelse(z[i], alpha[j], 1 - beta[j]))
    
    
    rating <- matrix(y, ncol = 1)
    item <- rep(1:I, J)
    annotator <- rep(1:J, each = I)
    
    sim.data <- data.frame(item, annotator, rating)
    sim.data.missing <- sim.data[-sample(1:nrow(sim.data), nrow(sim.data) * missing), ]
    N <- nrow(sim.data.missing)
    
    succes <- (max(sim.data.missing$item) & length(unique(sim.data.missing$item)) == I &
                 max(sim.data.missing$annotator) & length(unique(sim.data.missing$annotator)) == J) 
  }
  return(list(I = I,
              J = J,
              N = N,
              D = D,
              w0 = w0,
              w = w,
              x = x,
              alpha = alpha,
              beta = beta,
              z = z,
              sim.data = sim.data.missing))
  
}

sim.data <- data.simulation(I = 500, K = 2, J = 10, D = 20, missing  = .5)

w_init <- rnorm(20)
w0_init <- rnorm(1)
alpha_init <- rep(.8, 10)
beta_init <- rep(.8, 10)

init_fun <- function(n) {
  list(alpha = alpha_init,
       beta = beta_init,
       w = w_init,
       w0 = w0_init)
}

fit_DS_predictors <- sampling(object = model_predictors, data = list(J = sim.data$J,
                                                                     I = sim.data$I,
                                                                     D = sim.data$D,
                                                                     N = sim.data$N,
                                                                     x = sim.data$x,
                                                                     y = sim.data$sim.data$rating,
                                                                     ii = sim.data$sim.data$item,
                                                                     jj = sim.data$sim.data$annotator),
                              seed = 13031990, chains = 4, init = init_fun)




# same model blank predictors.
fit_DS_blank_predictors <- sampling(object = model_predictors, data = list(J = sim.data$J,
                                                                           I = sim.data$I,
                                                                           D = sim.data$D,
                                                                           N = sim.data$N,
                                                                           x = matrix(0, nrow = 500, ncol = 20),
                                                                           y = sim.data$sim.data$rating,
                                                                           ii = sim.data$sim.data$item,
                                                                           jj = sim.data$sim.data$annotator),
                                    seed = 13031990, chains = 4, init = init_fun)

z_hat <- ifelse(summary(fit_DS_blank_predictors, pars = "E_z")$summary[, 1] < 0.5, 0, 1)

fit.glm <- rstanarm::stan_glm(z_hat ~ 1 + sim.data$x, family = binomial())

# SSE_twostep[sim] <- sum((summary(fit.glm)[1:21,c(1)] - c(w0, w))^2)
# SSE_direct[sim] <- sum((summary(fit_DS_predictors, pars = c("w0", "w"))$summary[, c(1)] - c(w0, w))^2)




# }
# # for(sim in 1:nsim){
#   I <- 500
#   K <- 2
#   J <- 10
#   D <- 20
#   missing <- .5
#   
#   x <- matrix(rnorm(I * D), I, D)
#   w <- rnorm(D)
#   w0 <- -2
#   
#   alpha <- runif(J, 0.65, 0.95)
#   beta <- runif(J, 0.65, 0.95)
#   
#   z <- rep(0, I);
#   for (i in 1:I)
#     z[i] <- rbinom(1, 1, inv_logit(w0 + (x[i, ] %*% w)))
#   
#   y <- matrix(0, I, J);
#   for (i in 1:I)
#     for (j in 1:J)
#       y[i, j] <- rbinom(1, 1, ifelse(z[i], alpha[j], 1 - beta[j]))
#   
#   
#   rating <- matrix(y, ncol = 1)
#   item <- rep(1:I, J)
#   annotator <- rep(1:J, each = I)
#   
#   sim.data <- data.frame(item, annotator, rating)
#   sim.data.missing <- sim.data[-sample(1:nrow(sim.data), nrow(sim.data) * missing), ]
#   N <- nrow(sim.data.missing)
#   
#   (max(sim.data.missing$item) & length(unique(sim.data.missing$item)) == I &
#   max(sim.data.missing$annotator) & length(unique(sim.data.missing$annotator)) == J) 
#   
