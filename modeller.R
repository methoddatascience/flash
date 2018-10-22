library(tidyverse)
library(caret)
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
price_lm$xlevels[["Exterior1st"]] <- union(price_lm$xlevels[["Exterior1st"]], levels(validation$Exterior1st))
price_lm$xlevels[["Condition2"]] <- union(price_lm$xlevels[["Condition2"]], levels(validation$Condition1))
price_lm$xlevels[["Condition2"]] <- union(price_lm$xlevels[["Condition2"]], levels(validation$Exterior1st))

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