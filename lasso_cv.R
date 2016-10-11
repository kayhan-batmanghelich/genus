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

numOutCV = 5
numInCV = 4
for (x in 1:100) {
    #Flds <- createFolds(y, k=numOutCV, list=TRUE)
    trainIdx = createFolds(y, k = numOutCV, list = TRUE, returnTrain = TRUE)  # this will give you a training index
    # ************  WRITE A CODE HERE
    #     FOR LOOP OVER trainIdx and FOR EACH FOLD SUBTRACT FROM seq(length(y)) TO GET THE TEST INDICES
    #      something like this
    #   testIdx = list()
    #   for (i in seq(numOutCV)) {
    #        testIdx[ sprintf("Fold%01d", i) ] = setdiff ( seq(length(y)), trainIdx[ sprintf("Fold%01d", i)  ]  ) 
    #   }
    models_auc <- list()
    models_class <- list()
    auc_best_lambdas <- c()
    class_best_lambdas <- c()

    fold_vars <- sprintf("Flds$Fold%01d", seq(numOutCV))

    # compute models
    for (i in seq(numOutCV)) {
        # find the best lambda from the training data based on AUC
        #models_auc[[i]] <- cv.glmnet(X[eval(parse(text=fold_vars[i])),],
        #                 y[eval(parse(text=fold_vars[i]))],
        #                 family='binomial',
        #                 type.measure='auc',
        #                 nfolds=numInCV)
        models_auc[[i]] <- cv.glmnet(X[trainIdx['Fold%01d'],],
                         y[trainIdx['Fold%01d'],],
                         family='binomial',
                         type.measure='auc',
                         nfolds=numInCV)
        # find the best lambda
        auc_best_lambdas[i] <- models_auc[[i]]$lambda.min
        y_predic = predict(models_auc[[i]], type='class', newx = X[testIdx['Fold%01d'],] , s='lambda.min'  )
        # WRITE A CODE TO compare y_predic and y[testIdx['Fold%01d'],] to see how accurate it is  .....
        
        
        
       # find the best lambda from the training data based on ACCURACY -- i didn't write it but you got the idea
       models_class[[i]] <- cv.glmnet(X[eval(parse(text=fold_vars[i])),],
                         y[eval(parse(text=fold_vars[i]))],
                         family='binomial',
                         type.measure='class',
                         nfolds=numInCV)
    }

    # NONE OF THESE BELOW!
    ## get the bestscores and lambdas
    #for (i in seq(3)) {
    #  auc_best_lambdas[i] <- models_auc[[i]]$lambda.min
    #  class_best_lambdas[i] <- models_class[[i]]$lambda.min
    #}

    ## cross-validate on the left out fold using lambdas from the 3 models computed
    #mod_auc_4 <- cv.glmnet(X[Flds$Fold4,], y[Flds$Fold4], family='binomial', type.measure='auc', lambda=auc_best_lambdas)
    #mod_class_4 <- cv.glmnet(X[Flds$Fold4,], y[Flds$Fold4], family='binomial', type.measure='class', lambda=class_best_lambdas)

    ## final test set on the 5th fold
    #auc_pred <- predict(mod_auc_4, newx= X[Flds$Fold5,],  type="class")
    #auc_accs[x] <- sum(as.numeric(auc_pred) == y[Flds$Fold5]) / length(y[Flds$Fold5])

    #class_pred <- predict(mod_class_4, newx= X[Flds$Fold5,],  type="class")
    #class_accs[x] <- sum(as.numeric(class_pred) == y[Flds$Fold5]) / length(y[Flds$Fold5])
    
}


#> mean(auc_accs)
#[1] 0.6535266

#> mean(class_accs)
#[1] 0.6563768
