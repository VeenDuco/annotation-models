N <- 1000
# X <- mvtnorm::rmvnorm(n = N, mean = c(.5, 1.5, -1), sigma = diag(3))
X <- mvtnorm::rmvnorm(n = N, mean = c(0, 0, 0), sigma = diag(3))
B <- matrix(c(1, 2, -1), nrow = 3)
Z <- X %*% B
probs <- 1 / (1 + exp(-Z))
Y <- rbinom(n = N, size = 1, prob = probs)
fit <- glm(Y ~ 1 + X, family = binomial(link = "logit"))
fit
Zhat <- predict(fit) # log odds
# phat <- 1 / (1 + exp(-Zhat))
# logOdds <- log(phat / (1 - phat))
# all(round(logOdds,2) == round(Zhat,2))
train.id <- sample(1:N, 0.75 * N)

sim.data <- data.frame(X = X,
                       Y = Y)

test.data <- sim.data[-train.id, ]
train.data <- sim.data[train.id, ]
# Y.test <- Y[-train.id]
# X.test <- X[-train.id]
# Y.train <- Y[train.id]
# X.train <- X[train.id]
fit <- glm(Y ~ 1 + X.1 + X.2 + X.3, data = train.data, family = binomial(link = "logit"))
test.predict <- predict(fit, test.data, type = "response") # probs
test.predict[test.predict < .5] <- 0
test.predict[test.predict >= .5] <- 1
sum(test.predict == Y[-train.id]) / length(test.predict)

test.predict.logodds <- predict(fit, test.data) # logoddss
fit.lm <- lm(predict(fit, test.data) ~ test.data$X.1 + test.data$X.2 + test.data$X.3)
hist(predict(fit.lm))


## poly.
x <- sample( LETTERS[1:4], 10000, replace=TRUE, prob=c(0.1, 0.2, 0.65, 0.05) )
prop.table(table(x))


data
library(lme4)
glmer(rating ~ 1 + (1|item), family = binomial(), data = data)
data$rating <- as.factor(data$rating)
data$item  <- as.factor(data$item)
data$annotator <- as.factor(data$annotator)
# fit <- MASS::polr(rating ~ 1, data = data)
# fit <- MASS::polr(rating ~ 1 + item, data = data)
fit <- MASS::polr(rating ~ 1 + item + annotator, data = data)
predict(fit)
sum(predict(fit) == data$rating) / length(data$rating)
XX <- predict(fit, type = "probs")
lm()
