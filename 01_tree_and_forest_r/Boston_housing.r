library(randomForest)
library(rpart)
library(readr)
library(rsample)

compare_models <- function(prop = 0.7, ntree = 10) {
  data <- read_csv("./data/boston_housing.csv")
  set.seed(123)
  
  # 1. Split the data into training and test sets
  # We use the 'prop' argument to define the proportion
  data_split <- initial_split(data, prop = prop)
  train_data <- training(data_split)
  test_data  <- testing(data_split)

  set.seed(123)
  # 2. Fit the regression tree model
  # We predict 'medv' using all other variables (formula = medv ~ .)
  # We use method = "anova" because this is a regression task
  model_tree <- rpart(
    medv ~ .,
    data = train_data,
    method = "anova"
  )

  set.seed(123)
  # 3. Fit the random forest model
  # We use the 'ntree' argument passed to the function
  model_forest <- randomForest(
    medv ~ .,
    data = train_data,
    ntree = ntree
  )

  # 4. Make predictions on the test set for both models
  pred_tree <- predict(model_tree, test_data)
  pred_forest <- predict(model_forest, test_data)
  
  # Get the actual 'y' values from the test set
  actual_y <- test_data$medv
  
  # 5. Calculate MSE for both models
  # MSE = mean((actual - predicted)^2)
  mse_tree <- mean((actual_y - pred_tree)^2)
  mse_forest <- mean((actual_y - pred_forest)^2)

  # 6. Prepare the final return list
  
  # A list containing the train/test data frames
  data_list <- list(train = train_data, test = test_data)
  
  # A list containing the two trained models
  models_list <- list(model_tree = model_tree, model_forest = model_forest)
  
  # A numeric vector with the MSE errors
  errors_vector <- c(mse_tree, mse_forest)

  # Return the final list as specified in the requirements
  return(list(
    data = data_list,
    models = models_list,
    errors = errors_vector
  ))
}