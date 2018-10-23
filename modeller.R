library(tidyverse)
library(randomForest)
library(xgboost)
library(ModelMetrics)
library(h2o)

# Read updated train and test files
train <- read.csv("Literature Review/Kernels/train_2.csv")
test <- read.csv("Literature Review/Kernels/test_2.csv")

# Model partition
outcome <- train$SalePrice
partition <- createDataPartition(y=outcome,
                                 p=.5,
                                 list=F)
training <- train[partition,]
validation <- train[-partition,]


#-----------------------------------------
# A Linear Model
#-----------------------------------------
price_lm <- lm(SalePrice ~ ., data = na.omit(training))
summary(price_lm)

# Validation
prediction <- predict(price_lm, validation, type="response")
model_output <- cbind(validation, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

# Test with RMSE
rmse(model_output$log_SalePrice,model_output$log_prediction)

# Predict testing data
prediction <- predict(price_lm, test)

# Combine the prediction output with the rest of the set
model_output <- cbind(test, prediction) 

sub <- data.frame(Id = model_output$Id, SalePrice = exp(model_output$prediction))
write.csv(sub, file = "output/sub2.csv", row.names = FALSE, quote = FALSE)

#-----------------------------------------
# Random Forest
#-----------------------------------------

price_rf <- randomForest(SalePrice ~ ., data=training)
# Validation
prediction <- predict(price_rf, validation)
model_output <- cbind(validation, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

# Test with RMSE
rmse(model_output$log_SalePrice,model_output$log_prediction)

# Predict testing data
prediction <- predict(price_rf, test)

# Combine the prediction output with the rest of the set
model_output <- cbind(test, prediction) 

sub <- data.frame(Id = model_output$Id, SalePrice = exp(model_output$prediction))
write.csv(sub, file = "output/sub3.csv", row.names = FALSE, quote = FALSE)

#-----------------------------------------
# XGboost
#-----------------------------------------

#Create matrices from the data frames
train_matrix<- as.matrix(training, rownames.force=NA)
validation_matrix<- as.matrix(validation, rownames.force=NA)

#Turn the matrices into sparse matrices
train_sparse <- as(train_matrix, "sparseMatrix")
validate_sparse <- as(validation_matrix, "sparseMatrix")

#Convert to xgb.DMatrix format
train_xgb <- xgb.DMatrix(data = train_sparse, label = train_sparse[,"SalePrice"]) 

#Cross validate the model
cv.sparse <- xgb.cv(data = train_xgb,
                    nrounds = 600,
                    min_child_weight = 0,
                    max_depth = 10,
                    eta = 0.02,
                    subsample = .7,
                    colsample_bytree = .7,
                    booster = "gbtree",
                    eval_metric = "rmse",
                    verbose = TRUE,
                    print_every_n = 50,
                    nfold = 4,
                    nthread = 2,
                    objective="reg:linear")

#Train the model

#Choose the parameters for the model
param <- list(colsample_bytree = .7,
              subsample = .7,
              booster = "gbtree",
              max_depth = 10,
              eta = 0.02,
              eval_metric = "rmse",
              objective="reg:linear")


#Train the model using those parameters
bstSparse <-
  xgb.train(params = param,
            data = train_xgb,
            nrounds = 600,
            watchlist = list(train = train_xgb),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 2)

validation_xgb <- xgb.DMatrix(data = validate_sparse)

#Make the prediction based on the half of the training data set aside
prediction <- predict(bstSparse, validate_sparse) 

#Put testing prediction and test dataset all together
validation_2 <- as.data.frame(as.matrix(validate_sparse))
prediction <- as.data.frame(as.matrix(prediction))
colnames(prediction) <- "prediction"
model_output <- cbind(validation_2, prediction)

#Test with RMSE
rmse(model_output$SalePrice,model_output$prediction)

rm(bstSparse)

#Create matrices from the data frames
retrain_data<- as.matrix(train, rownames.force=NA)

#Turn the matrices into sparse matrices
retrain <- as(retrain_data, "sparseMatrix")

param <- list(colsample_bytree = .7,
              subsample = .7,
              booster = "gbtree",
              max_depth = 10,
              eta = 0.02,
              eval_metric = "rmse",
              objective="reg:linear")

retrain_xgb <- xgb.DMatrix(data = retrain, label = retrain[,"SalePrice"])

#retrain the model using those parameters
bstSparse <-
  xgb.train(params = param,
            data = retrain_xgb,
            nrounds = 600,
            watchlist = list(train = train_xgb),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 2)

# Get the supplied test data ready #
#Get the dataset formatted as a frame for later combining
test$SalePrice <- 0
predict <- as.data.frame(test[, -c(11)])

#Create matrices from the data frames
pred_data<- as.matrix(predict, rownames.force=NA)

#Turn the matrices into sparse matrices
predicting <- as(pred_data, "sparseMatrix")

prediction <- predict(bstSparse, predicting)

#Get the dataset formatted as a frame for later combining
prediction <- as.data.frame(as.matrix(prediction))  

colnames(prediction) <- "prediction"

#Combine the prediction output with the rest of the set
model_output <- cbind(predict, prediction) 

sub <- data.frame(Id = test$Id, SalePrice = exp(model_output$prediction))
write.csv(sub, file = "output/sub4.csv", row.names = FALSE, quote = FALSE)

########################################
## Feature Selection
########################################
base.mod <- lm( SalePrice ~ 1 , data = train)  # base intercept only model
all.mod <- lm( SalePrice ~ . ,
               data = train) # full model with all predictors
stepMod <-
  step(
    base.mod,
    scope = list(lower = base.mod, upper = all.mod),
    direction = "both",
    trace = 0,
    steps = 1000
  )  # perform step-wise algorithm
shortlistedVars <-
  names(unlist(stepMod[[1]])) # get the shortlisted variable.
shortlistedVars <-
  shortlistedVars[!shortlistedVars %in% "(Intercept)"]  # remove intercept
print(shortlistedVars)
summary(stepMod)

y.dep <- grep("SalePrice", colnames(train))
x.indep <- c(grep("OverallQual", colnames(train)), 
             grep("TotalSF", colnames(train)),
             grep("GarageCars", colnames(train)),
             grep("YearRemodAdd", colnames(train)),
             grep("Fireplaces", colnames(train)),
             grep("YearBuilt", colnames(train)),
             grep("GarageType_1", colnames(train)),
             grep("HeatingQC", colnames(train)),
             grep("GarageType_1", colnames(train)),
             grep("KitchenQual", colnames(train)),
             grep("ExterQual", colnames(train)))

h2o.init()
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)
r <- h2o.runif(train.h2o)
trainHex.split <- h2o.splitFrame(train.h2o, ratios=.70)
#--------------------------------
# H2O Multiple Regression 
#--------------------------------

# Base GLM
regression.model <- h2o.glm( y = y.dep,
                             x = x.indep,
                             training_frame = trainHex.split[[1]],
                             nfolds =  10,
                             validation_frame = trainHex.split[[2]],
                             family = "gaussian",
                             seed = 123)
h2o.performance(regression.model)

predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub <- data.frame(Id = test$Id, SalePrice  = exp(predict.reg$predict))
write.csv(sub, file = "output/sub5.csv", row.names = FALSE, quote = FALSE)

# Grid Search
solvers <- c("IRLSM", "L_BFGS", "COORDINATE_DESCENT_NAIVE", "COORDINATE_DESCENT")

families <- c("gaussian", "poisson", "gamma")

gaussianLinks <- c("identity", "log", "inverse")

poissonLinks <- c("log")

gammaLinks <- c("identity", "log", "inverse")

gammaLinks_CD <- c("identity", "log")

allGrids <- lapply(solvers, function(solver){
  lapply(families, function(family){
    
    if(family == "gaussian")theLinks <- gaussianLinks
    else if(family == "poisson")theLinks <- poissonLinks
    else{
      if(solver == "COORDINATE_DESCENT")theLinks <- gammaLinks_CD
      else theLinks = gammaLinks
    }
    
    lapply(theLinks, function(link){
      grid_id = paste("GLM", solver, family, link, sep="_")
      h2o.grid("glm", grid_id = grid_id,
               hyper_params = list(
                 alpha = c(0, 0.1, 0.5, 0.99)
               ),
               x = x.indep,
               y = y.dep,
               training_frame =  trainHex.split[[1]],
               validation_frame = trainHex.split[[2]],
               nfolds = 10,
               lambda_search = TRUE,
               solver = solver,
               family = family,
               link = link,
               max_iterations = 1000
      )
    })
  })
})

get_best_id <- function(grid) {
  # print out the mse for all of the models
  model_ids <- grid@model_ids
  rmse <- vector(mode="numeric", length=0)
  grid_models <- lapply(model_ids, function(model_id) { model = h2o.getModel(model_id) })
  for (i in 1:length(grid_models)) {
    print(sprintf("rmse: %f", h2o.rmse(grid_models[[i]])))
    rmse[i] <- h2o.rmse(grid_models[[i]])
  }
  
  best_id <- model_ids[order(rmse,decreasing=F)][1]
  print(best_id)
  best_id
}

grids_list <- unlist(allGrids)
all_models <- map(grids_list, ~ get_best_id(.x))



fit.best <- h2o.getModel(model_id = "GLM_COORDINATE_DESCENT_gaussian_identity_model_3")
h2o.performance(fit.best)

predict.glm <- as.data.frame(h2o.predict(fit.best, test.h2o))
sub <- data.frame(Id = test$Id, SalePrice  = exp(predict.glm$predict))
write.csv(sub, file = "output/sub6.csv", row.names = FALSE, quote = FALSE)

#--------------------------------
# H2O Random Forest
#--------------------------------
grid <- h2o.grid("randomForest",
                 hyper_params = list(
                   ntrees = c(70),
                   #mtries = c(2, 3, 4, 5),
                   max_depth=c(70),
                   sample_rate = c(0.5),
                   col_sample_rate_per_tree = c(0.5, 0.7, 0.9, 1.0)
                 ),
                 y = y.dep, 
                 x = x.indep,
                 seed = 123,
                 training_frame = trainHex.split[[1]],
                 validation_frame = trainHex.split[[2]],
                 nfolds = 10,
                 stopping_metric = "RMSE",
                 stopping_tolerance = 0,
                 stopping_rounds = 4,
                 score_tree_interval = 3)

# print out all prediction errors and run times of the models
grid

# print out the mse for all of the models
model_ids <- grid@model_ids
rmse <- vector(mode="numeric", length=0)
grid_models <- lapply(model_ids, function(model_id) { model = h2o.getModel(model_id) })
for (i in 1:length(grid_models)) {
  print(sprintf("rmse: %f", h2o.rmse(grid_models[[i]])))
  rmse[i] <- h2o.rmse(grid_models[[i]])
}

best_id <- model_ids[order(rmse,decreasing=F)][1]
best_id

fit.best <- h2o.getModel(model_id = best_id[[1]])
h2o.performance(fit.best)

predict.rf <- as.data.frame(h2o.predict(fit.best, test.h2o))
sub <- data.frame(Id = test$Id, SalePrice  = exp(predict.rf$predict))
write.csv(sub, file = "output/sub12.csv", row.names = FALSE, quote = FALSE)

#--------------------------------
# H2O GBM
#--------------------------------
search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 10, 
                        seed = 1)

grid <- h2o.grid("gbm",grid_id = "my_grid",
                 search_criteria = search_criteria,
                 hyper_params = list(
                   max_depth = c(20, 30),
                   #min_rows = c(2, 5, 10),
                   sample_rate = c(0.5, 0.8), #0.95, 1.0),
                   ##col_sample_rate = c(0.5, 0.8),# 0.95, 1.0),
                   #col_sample_rate_per_tree = c(0.8, 0.99, 1.0),
                   learn_rate = seq(0.0001, 0.2, 0.0001),
                   ntrees = c(100, 150),# 200, 250),
                   seed = 123
                 ),
                 y = y.dep, x = x.indep,
                 # stopping_tolerance = 0.001,
                 # stopping_rounds=3,
                 # score_tree_interval = 10,
                 distribution="gaussian",
                 training_frame = trainHex.split[[1]],
                 validation_frame = trainHex.split[[2]])
grid

# print out the mse for all of the models
model_ids <- grid@model_ids
rmse <- vector(mode="numeric", length=0)
grid_models <- lapply(model_ids, function(model_id) { model = h2o.getModel(model_id) })
for (i in 1:length(grid_models)) {
  print(sprintf("rmse: %f", h2o.rmse(grid_models[[i]])))
  rmse[i] <- h2o.rmse(grid_models[[i]])
}


best_id <- model_ids[order(rmse,decreasing=F)][1]
best_id

fit.best <- h2o.getModel(model_id = best_id[[1]])
h2o.performance(fit.best)

predict.gbm <- as.data.frame(h2o.predict(fit.best, test.h2o))
sub <- data.frame(Id = test$Id, SalePrice  = exp(predict.gbm$predict))
write.csv(sub, file = "output/sub13.csv", row.names = FALSE, quote = FALSE)

# #--------------------------------
# # H2O XGBoost
# #--------------------------------
# grid <- h2o.xgboost(
#   x = x.indep,
#   y = y.dep,
#   training_frame = trainHex.split[[1]],
#   validation_frame = trainHex.split[[2]],
#   nfolds = 10,
#   stopping_metric = "RMSE",
#   learn_rate = seq(0.0001, 0.2, 0.0001),
#   ntrees = c(100, 150),# 200, 250),
#   max_depth = c(20, 30),
#   seed = 123
# )
# grid
# 
# # print out the mse for all of the models
# model_ids <- grid@model_ids
# rmse <- vector(mode="numeric", length=0)
# grid_models <- lapply(model_ids, function(model_id) { model = h2o.getModel(model_id) })
# for (i in 1:length(grid_models)) {
#   print(sprintf("rmse: %f", h2o.rmse(grid_models[[i]])))
#   rmse[i] <- h2o.rmse(grid_models[[i]])
# }
# 
# 
# best_id <- model_ids[order(rmse,decreasing=F)][1]
# best_id
# 
# fit.best <- h2o.getModel(model_id = best_id[[1]])
# h2o.performance(fit.best)
# 
# predict.gbm <- as.data.frame(h2o.predict(fit.best, test.h2o))
# sub <- data.frame(Id = test$Id, SalePrice  = exp(predict.gbm$predict))
write.csv(sub, file = "output/sub13.csv", row.names = FALSE, quote = FALSE)