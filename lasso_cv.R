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
}

# find lambda.min that corresponds to the best score
auc_score <- max(auc_best_scores)
auc_position <- c()
for (i in 1:4){
    try(auc_position[[1]] <- which(models_auc[[i]]$cvm %in% auc_score), silent=TRUE)
}
