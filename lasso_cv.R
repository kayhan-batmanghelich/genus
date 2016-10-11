setwd('/data/petryshen/yoel/lasso')
require(caret)
require(glmnet)

X <- read.csv('endo_mat_1035_170.csv')
X <- as.matrix(sapply(X, as.matrix))
y <- read.csv('reduced_y.txt', header=FALSE)
y <- as.vector(as.matrix(y))

if (dim(X)[1] != length(y)) {
    for (i in 1:10) {
        print("X and y do not have the same number of samples")
    }
} else {
    print("X and y have the same number of samples")
}

Flds <- createFolds(y, k=4, list=TRUE)
models_auc <- list()
models_class <- list()
auc_best_scores <- c()
class_best_scores <- c()
auc_best_lambdas <- c()
class_best_lambdas <- c()

fold_vars <- sprintf("Flds$Fold%01d", seq(4))

# compute models
for (i in seq(4)) {
    models_auc[[i]] <- cv.glmnet(X[eval(parse(text=fold_vars[i])),],
                     y[eval(parse(text=fold_vars[i]))],
                     family='binomial',
                     type.measure='auc',
                     nfolds=4)

   models_class[[i]] <- cv.glmnet(X[eval(parse(text=fold_vars[i])),],
                     y[eval(parse(text=fold_vars[i]))],
                     family='binomial',
                     type.measure='class',
                     nfolds=4)
}  

# get the best scores of each model
for (i in seq(4)) {                 
  auc_best_scores[i] <- max(models_auc[[i]]$cvm)
  class_best_scores[i] <- max(models_class[[i]]$cvm)
  auc_best_lambdas[i] <- models_auc[[i]]$lambda.min
  class_best_lambdas[i] <- models_class[[i]]$lambda.min
}

# find lambda.min that corresponds to the best score
auc_score <- max(auc_best_scores)
auc_position <- which.max(auc_best_scores)
auc_lambda <- models_auc[[auc_position]]$lambda.min
class_score <- max(class_best_scores)
class_position <- which.max(class_best_scores)
class_lambda <- models_class[[class_position]]$lambda.min

# run final models with cv using best lambdas
aucCV <- cv.glmnet(X,y, family='binomial', type.measure='auc', lambda=auc_best_lambdas)
classCV <- cv.glmnet(X,y, family='binomial', type.measure='class', lambda=class_best_lambdas)
