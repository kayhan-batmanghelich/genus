setwd('/data/petryshen/yoel/lasso')
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

auc_accs <- c()
class_accs <- c()

for (i in 1:30) {
    Flds <- createFolds(y, k=5, list=TRUE)
    models_auc <- list()
    models_class <- list()
    auc_best_lambdas <- c()
    class_best_lambdas <- c()

    fold_vars <- sprintf("Flds$Fold%01d", seq(5))

    # compute models
    for (i in seq(3)) {
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

    # get the bestscores and lambdas
    for (i in seq(3)) {
      auc_best_lambdas[i] <- models_auc[[i]]$lambda.min
      class_best_lambdas[i] <- models_class[[i]]$lambda.min
    }

    # cross-validate on the left out fold using lambdas from 4 models computed
    mod_auc_4 <- cv.glmnet(X[Flds$Fold4,], y[Flds$Fold4], family='binomial', type.measure='auc', lambda=auc_best_lambdas)
    mod_class_4 <- cv.glmnet(X[Flds$Fold4,], y[Flds$Fold4], family='binomial', type.measure='class', lambda=class_best_lambdas)

    # final test set
    auc_pred <- predict(mod_auc_4, newx= X[Flds$Fold5,],  type="class")
    auc_accs[i] <- sum(as.numeric(auc_pred) == y[Flds$Fold5]) / length(y[Flds$Fold5])

    class_pred <- predict(mod_class_4, newx= X[Flds$Fold5,],  type="class")
    class_accs[i] <- sum(as.numeric(class_pred) == y[Flds$Fold5]) / length(y[Flds$Fold5])
    
}
