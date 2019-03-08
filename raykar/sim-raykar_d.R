library(rstan)
inv_logit <- function(u) 1 / (1 + exp(-u))

set.seed(13031990)

I <- 500
K <- 2
J <- 10
D <- 20
missing <- .3

x <- matrix(rnorm(N * D), N, D)
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


# model no predictors
model_no_predictors <- stan_model("DS_no_predictors.stan")

pi_init <- table(sim.data[, 3]) / sum(table(sim.data[, 3]))
beta_init <- array(NA, dim = c(J, K, K))
for(j in 1:J){
  for(k in 1:K){
    beta_init[j, k, ] <- rep(0.2 / (K - 1), K)
    beta_init[j , k, k] <- 0.8
  }
}


init_fun <- function(n) {
  list(pi = pi_init,
       beta = beta_init)
}

fit_DS_no_predictors <- sampling(object = model_no_predictors, data = list(J  = J,
                                                                           K  = K,
                                                                           N  = N,
                                                                           I  = I,
                                                                           ii = sim.data.missing$item,
                                                                           jj = sim.data.missing$annotator,
                                                                           y  = sim.data.missing$rating + 1), # +1 due to binairy 01 to 12
                                 seed = 13031990, chains = 4, init = init_fun)

summary(fit_DS_no_predictors, pars = "q_z")$summary


model_predictors <- stan_model("raykar/raykar_marginal_wide.stan")

w_init <- rnorm(D)
w0_init <- rnorm(1)
alpha_init <- rep(.8, J)
beta_init <- rep(.8, J)

init_fun <- function(n) {
  list(alpha = alpha_init,
       beta = beta_init,
       w = w_init,
       w0 = w0_init)
}

fit_DS_predictors <- sampling(object = model_predictors, data = list(J = J,
                                                                     I = I,
                                                                     D = D,
                                                                     x = x,
                                                                     y = y), # +1 due to binairy 01 to 12
                              seed = 13031990, chains = 4, init = init_fun)

true.pred.in.95CI <- summary(fit_DS_predictors, pars = "w")$summary[, c(4)] < w & summary(fit_DS_predictors, pars = "w")$summary[, c(8)] > w
sum(true.pred.in.95CI) / length(w)

true.sens.in.95CI <- summary(fit_DS_predictors, pars = "alpha")$summary[, c(4)] < alpha & summary(fit_DS_predictors, pars = "alpha")$summary[, c(8)] > alpha
sum(true.sens.in.95CI) / length(alpha)

true.spec.in.95CI <- summary(fit_DS_predictors, pars = "beta")$summary[, c(4)] < beta & summary(fit_DS_predictors, pars = "beta")$summary[, c(8)] > beta
sum(true.spec.in.95CI) / length(beta)

z_hat <- ifelse(summary(fit_DS_predictors, pars = "E_z")$summary[, 1] < 0.5, 0, 1)
sum(z_hat == z) / length(z)




# same model blank predictors.
fit_DS_blank_predictors <- sampling(object = model_predictors, data = list(J = J,
                                                                           I = I,
                                                                           D = D,
                                                                           x = matrix(0, nrow = 500, ncol = 20),
                                                                           y = y), 
                                    seed = 13031990, chains = 4, init = init_fun)

true.pred.in.95CI <- summary(fit_DS_blank_predictors, pars = "w")$summary[, c(4)] < w & summary(fit_DS_blank_predictors, pars = "w")$summary[, c(8)] > w
sum(true.pred.in.95CI) / length(w)

true.sens.in.95CI <- summary(fit_DS_blank_predictors, pars = "alpha")$summary[, c(4)] < alpha & summary(fit_DS_blank_predictors, pars = "alpha")$summary[, c(8)] > alpha
sum(true.sens.in.95CI) / length(alpha)

true.spec.in.95CI <- summary(fit_DS_blank_predictors, pars = "beta")$summary[, c(4)] < beta & summary(fit_DS_blank_predictors, pars = "beta")$summary[, c(8)] > beta
sum(true.spec.in.95CI) / length(beta)

z_hat <- ifelse(summary(fit_DS_blank_predictors, pars = "E_z")$summary[, 1] < 0.5, 0, 1)
sum(z_hat == z) / length(z)

fit.glm <- rstanarm::stan_glm(z_hat ~ 1 + x_save, family = binomial())

sum((summary(fit.glm)[1:21,c(1)] - c(w0, w))^2)
sum((summary(fit_DS_predictors, pars = c("w0", "w"))$summary[, c(1)] - c(w0, w))^2)

# model wide raykar no predictors
# model predictors
model_no_predictors_raykar <- stan_model("raykar/raykar_marginal_wide_no_predictors.stan")

w0_init <- rnorm(1)
alpha_init <- rep(.8, J)
beta_init <- rep(.8, J)

init_fun <- function(n) {
  list(alpha = alpha_init,
       beta = beta_init,
       w0 = w0_init)
}

fit_DS_no_predictors_raykar <- sampling(object = model_no_predictors_raykar, data = list(J = J,
                                                                     I = I,
                                                                     y = y), # +1 due to binairy 01 to 12
                              seed = 13031990, chains = 4, init = init_fun)

# true.pred.in.95CI <- summary(fit_DS_predictors, pars = "w")$summary[, c(4)] < w & summary(fit_DS_predictors, pars = "w")$summary[, c(8)] > w
# sum(true.pred.in.95CI) / length(w)

true.sens.in.95CI <- summary(fit_DS_no_predictors_raykar, pars = "alpha")$summary[, c(4)] < alpha & summary(fit_DS_no_predictors_raykar, pars = "alpha")$summary[, c(8)] > alpha
sum(true.sens.in.95CI) / length(alpha)

true.spec.in.95CI <- summary(fit_DS_no_predictors_raykar, pars = "beta")$summary[, c(4)] < beta & summary(fit_DS_no_predictors_raykar, pars = "beta")$summary[, c(8)] > beta
sum(true.spec.in.95CI) / length(beta)

z_hat <- ifelse(summary(fit_DS_no_predictors_raykar, pars = "E_z")$summary[, 1] < 0.5, 0, 1)
sum(z_hat == z) / length(z)

fitglm <- glm(z_hat ~ 1 + x, family = binomial())
summary(fitglm)$coef[, 1] - 1.96 * summary(fitglm)$coef[, 2] < c(w0, w) & summary(fitglm)$coef[, 1] + 1.96 * summary(fitglm)$coef[, 2] > c(w0, w)



#### 
round(data.frame("simulated value" = c(w0, w),
           "full_raykar_mean" = summary(fit_DS_predictors, pars = c("w0", "w"))$summary[, 1],
           "two_step_mean" = summary(fitglm)$coef[, 1],
           "full_raykar_sd" = summary(fit_DS_predictors, pars = c("w0", "w"))$summary[, 3],
           "two_step_sd" = summary(fitglm)$coef[, 2]), 2)
