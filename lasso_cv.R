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

# performs stratified k fold 
F <- createFolds(y, k=10, list=TRUE)
