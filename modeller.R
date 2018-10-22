library(tidyverse)
library(randomForest)
library(xgboost)
library(ModelMetrics)

# Read updated train and test files
train <- read.csv("Literature Review/Kernels/train_1.csv")
test <- read.csv("Literature Review/Kernels/test_1.csv")

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
