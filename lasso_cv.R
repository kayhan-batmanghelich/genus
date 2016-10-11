setwd('/Users/yoelsanchez/test')
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

# stratified k fold
F <- createFolds(y, k=5, list=TRUE)

# list to hold everything in
models <- list()

# format string
fold_vars <- sprintf("F$Fold%01d", seq(5))

for (i in seq(5)) {
    mod <- cv.glmnet(X[eval(parse(text=fold_vars[i])),],
                     y[eval(parse(text=fold_vars[i]))],
                     family='binomial',
                     type.measure='auc',
                     n_folds=5)

}
