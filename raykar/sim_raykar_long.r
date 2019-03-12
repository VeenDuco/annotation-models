# settings
library(rstan)
# options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# model_predictors <- stan_model("raykar/raykar_marginal_long.stan")
# saveRDS(model_predictors, "raykar/raykar_marginal_long.RDS")
model_predictors <- readRDS("raykar/raykar_marginal_long.RDS")

# define functions
inv_logit <- function(u) 1 / (1 + exp(-u))
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

# set n simulations
nsim <- 10
J = 10
I = 500
K = 2
D = 20
missing = .5
nchains = 4

# create output for simulations
predictors.overview <- list(
  simulated.predictors = matrix(NA, nrow = 21, ncol = nsim),
  direct_estimated = matrix(NA, nrow = 21, ncol = nsim),
  two_step_estimated = matrix(NA, nrow = 21, ncol = nsim)
)

annotator.overview <- list(
  true_anno_sens = matrix(NA, nrow = J, ncol = nsim),
  direct_lb_sens = matrix(NA, nrow = J, ncol = nsim),
  direct_mean_sens = matrix(NA, nrow = J, ncol = nsim),
  direct_ub_sens = matrix(NA, nrow = J, ncol = nsim),
  two_step_lb_sens = matrix(NA, nrow = J, ncol = nsim),
  two_step_mean_sens = matrix(NA, nrow = J, ncol = nsim),
  two_step_ub_sens = matrix(NA, nrow = J, ncol = nsim),
  true_anno_spec = matrix(NA, nrow = J, ncol = nsim),
  direct_lb_spec = matrix(NA, nrow = J, ncol = nsim),
  direct_mean_spec = matrix(NA, nrow = J, ncol = nsim),
  direct_ub_spec = matrix(NA, nrow = J, ncol = nsim),
  two_step_lb_spec = matrix(NA, nrow = J, ncol = nsim),
  two_step_mean_spec = matrix(NA, nrow = J, ncol = nsim),
  two_step_ub_spec = matrix(NA, nrow = J, ncol = nsim)
)

z_overview <- list(
  z_sim = matrix(NA, nrow = I, ncol = nsim),
  direct_z_hat = matrix(NA, nrow = I, ncol = nsim),
  two_step_z_hat = matrix(NA, nrow = I, ncol = nsim)
)


set.seed(13031990)
for(sim in 1:nsim){
print(paste0("Start of simulation ", sim, "out of ", nsim, "."))  
sim.data <- data.simulation(I = I, K = K, J = J, D = D, missing  = missing)

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

fit_DS_predictors <- sampling(object = model_predictors, data = list(J = sim.data$J,
                                                                     I = sim.data$I,
                                                                     D = sim.data$D,
                                                                     N = sim.data$N,
                                                                     x = sim.data$x,
                                                                     y = sim.data$sim.data$rating,
                                                                     ii = sim.data$sim.data$item,
                                                                     jj = sim.data$sim.data$annotator),
                              seed = 13031990, chains = nchains, init = init_fun)


print(paste0("Halfway of simulation ", sim, "out of ", nsim, "."))
# same model blank predictors.
fit_DS_blank_predictors <- sampling(object = model_predictors, data = list(J = sim.data$J,
                                                                           I = sim.data$I,
                                                                           D = sim.data$D,
                                                                           N = sim.data$N,
                                                                           x = matrix(0, nrow = I, ncol = D),
                                                                           y = sim.data$sim.data$rating,
                                                                           ii = sim.data$sim.data$item,
                                                                           jj = sim.data$sim.data$annotator),
                                    seed = 13031990, chains = nchains, init = init_fun)

z_hat <- ifelse(summary(fit_DS_blank_predictors, pars = "E_z")$summary[, 1] < 0.5, 0, 1)

fit.glm <- rstanarm::stan_glm(z_hat ~ 1 + sim.data$x, family = binomial())

# predictor output
predictors.overview$simulated.predictors[, sim] <- c(sim.data$w0, sim.data$w)
predictors.overview$direct_estimated[, sim] <- summary(fit_DS_predictors, pars = c("w0", "w"))$summary[, 1]
predictors.overview$two_step_estimated[, sim] <- summary(fit.glm)[1:21, 1]


annotator.overview$true_anno_sens[, sim] <- sim.data$alpha
annotator.overview$direct_lb_sens[, sim] <- summary(fit_DS_predictors, pars = c("alpha"))$summary[, 4]
annotator.overview$direct_mean_sens[, sim] <- summary(fit_DS_predictors, pars = c("alpha"))$summary[, 1]
annotator.overview$direct_ub_sens[, sim] <- summary(fit_DS_predictors, pars = c("alpha"))$summary[, 8]
annotator.overview$two_step_lb_sens[, sim] <- summary(fit_DS_blank_predictors, pars = c("alpha"))$summary[, 4]
annotator.overview$two_step_mean_sens[, sim] <- summary(fit_DS_blank_predictors, pars = c("alpha"))$summary[, 1]
annotator.overview$two_step_ub_sens[, sim] <- summary(fit_DS_blank_predictors, pars = c("alpha"))$summary[, 8]
annotator.overview$true_anno_spec[, sim] <- sim.data$beta
annotator.overview$direct_lb_spec[, sim] <- summary(fit_DS_predictors, pars = c("beta"))$summary[, 4]
annotator.overview$direct_mean_spec[, sim] <- summary(fit_DS_predictors, pars = c("beta"))$summary[, 1]
annotator.overview$direct_ub_spec[, sim] <- summary(fit_DS_predictors, pars = c("beta"))$summary[, 8]
annotator.overview$two_step_lb_spec[, sim] <- summary(fit_DS_blank_predictors, pars = c("beta"))$summary[, 4]
annotator.overview$two_step_mean_spec[, sim] <- summary(fit_DS_blank_predictors, pars = c("beta"))$summary[, 1]
annotator.overview$two_step_ub_spec[, sim] <- summary(fit_DS_blank_predictors, pars = c("beta"))$summary[, 8]
                         
z_overview$z_sim[, sim] <- sim.data$z
z_overview$direct_z_hat[, sim] <- ifelse(summary(fit_DS_predictors, pars = "E_z")$summary[, 1] < 0.5, 0, 1)
z_overview$two_step_z_hat[, sim] <- ifelse(summary(fit_DS_blank_predictors, pars = "E_z")$summary[, 1] < 0.5, 0, 1)

print(paste0("End of simulation ", sim, "out of ", nsim, "."))
}
rm(fit_DS_blank_predictors)
rm(fit_DS_predictors)
rm(fit.glm)
save.image("raykar/simulation_long_raykar_nsim10.rdata")

sum(z_overview$z_sim == z_overview$direct_z_hat)
sum(z_overview$z_sim == z_overview$two_step_z_hat)
rowMeans((predictors.overview$simulated.predictors - predictors.overview$direct_estimated))
rowMeans((predictors.overview$simulated.predictors - predictors.overview$two_step_estimated))
rowMeans(annotator.overview$true_anno_sens - annotator.overview$direct_mean_sens)
rowMeans(annotator.overview$true_anno_sens - annotator.overview$two_step_mean_sens)
rowMeans(annotator.overview$true_anno_spec - annotator.overview$direct_mean_spec)
rowMeans(annotator.overview$true_anno_spec - annotator.overview$two_step_mean_spec)
    
# bias_direct <- c(sim.data$w0, sim.data$w) - summary(fit_DS_predictors, pars = c("w0", "w"))$summary[, 1]
# bias_two_step <- c(sim.data$w0, sim.data$w) - summary(fit.glm)[1:21, 1]
# 
# 
# plot(bias_direct, bias_two_step, xlim = c(-2, 2), ylim = c(-2, 2))
# abline(a = 0, b = 1)
# abline(a = 0, b = -1)
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
