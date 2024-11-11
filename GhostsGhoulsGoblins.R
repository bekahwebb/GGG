library(tidyverse)
library(DataExplorer)
library(vroom)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(readr)
library(GGally)
library(poissonreg)
library(recipes)
library(rsample)
library(magrittr)
library(tidymodels)
library(lubridate)
library(poissonreg) #if you want to do penalized, poisson regression
library(rpart)
library(ranger)
library(stacks) # you need this library to create a stacked model
library(embed) # for target encoding
library(ggmosaic)
library(vroom)
library(embed)
library(themis)
library(parsnip)
library(discrim)
library('naivebayes')
library(bonsai)
library(lightgbm)
library("dbarts")

ggg_test <- read_csv('test.csv')
ggg_train <- read_csv('train.csv')
ggg_trainNA <- read_csv('trainWithMissingValues.csv') 
# ggg_train
# ggg_trainNA

#factor type and color for na data
ggg_trainNA$color = factor(ggg_trainNA$color)
ggg_trainNA$type = factor(ggg_trainNA$type)

#eda
ggplot(data=ggg_train, aes(x=type, y=bone_length)) +
geom_boxplot()

ggplot(data=ggg_train) + geom_mosaic(aes(x=product(color), fill=type))

#recipe
ggg_recipe <- recipe(type~., data = ggg_trainNA) %>% 
  step_impute_median(all_numeric_predictors())

prep <- prep(ggg_recipe) 
baked <- bake(prep, new_data = ggg_trainNA)  

#calculate the rmse
rmse_vec(ggg_train[is.na(ggg_trainNA)], baked[is.na(ggg_trainNA)])
#0.1520589


#try a random forest model for this dataset 11/6/24

#my recipe
# Feature Engineering
ggg_recipe <- recipe(type~., data=ggg_train) %>%
  step_dummy(color) %>% 
  step_range(all_numeric_predictors(), min = 0, max = 1)

rf_ggg_model <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=999) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Create a workflow with model & recipe

rf_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(rf_ggg_model)

## Set up grid of tuning values
rf_tunegrid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = 3) 

## Set up K-fold CV
folds <- vfold_cv(ggg_train, v = 10, repeats=1)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=rf_tunegrid,
            metrics=metric_set(accuracy)) 

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "accuracy") 

## Finalize workflow and predict
final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

## Predict
rf_ggg_predictions <- final_wf %>%
  predict(new_data =ggg_test, type="class")


## Format the Predictions for Submission to Kaggle
rf_ggg_kaggle_submission <- rf_ggg_predictions%>%
  rename(type=.pred_class) %>%
  bind_cols(., ggg_test) %>% #Bind predictions with test data
  select(id, type)  #keep Id, type for submission


## Write out the file
vroom_write(x=rf_ggg_kaggle_submission, file="rfGGGPreds.csv", delim=",")
#public score of .71455 with 500 trees
#try 900 trees, improved to .72022
#try 1100 trees, went back down to .71455
# try 999 trees, public score .71833
#.729 svm score increased some of the tuning paramaters .732
#.681 knn
#.747 is the cutoff
#.750 let's go! with naive bayes, improved from .72022 with making id as a feature


#my recipe
# Feature Engineering naive bayes
ggg_recipe <- recipe(type~., data = ggg_train) %>% 
  step_mutate(color = as.factor(color)) %>% 
  step_mutate(id, features = id) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)


ggg_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes

## Create a workflow with model & recipe

ggg_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(ggg_model)

## Set up grid of tuning values
tuning_grid <- grid_regular(Laplace(), smoothness(),levels = 3)

## Set up K-fold CV
folds <- vfold_cv(ggg_train, v = 10, repeats=1)

## Run the CV
CV_results <- ggg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) 

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="roc_auc")

## Finalize workflow and predict
final_wf <-
  ggg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

## Predict
ggg_predictions <- final_wf %>%
  predict(new_data =ggg_test, type="class")


## Format the Predictions for Submission to Kaggle
naivebayes_ggg_kaggle_submission <- ggg_predictions%>%
  rename(type=.pred_class) %>%
  bind_cols(., ggg_test) %>% #Bind predictions with test data
  select(id, type)  #keep Id, type for submission


## Write out the file
vroom_write(x=naivebayes_ggg_kaggle_submission, file="nbGGGPreds.csv", delim=",")
#public score 0.75047, done, the cutoff is .747

#neural networks 11/8/24
nn_recipe <- recipe(type~., data = ggg_train) %>%
  step_mutate(type = as.factor(type), skip = TRUE) %>%
  step_mutate(color = as.factor(color)) %>% 
  step_dummy(color) %>% # Turn color to factor then dummy encode color
  step_mutate(id, features = id) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 100 #or 100 or 250
) %>%
set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

#set workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)
nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)),
                            levels=3)

# Set up k-fold cross validation and run it
nn_folds <- vfold_cv(ggg_train, v = 5, repeats = 1)
CV_nn_results <- nn_wf %>%
  tune_grid(resamples = nn_folds,
          grid = nn_tuneGrid,
          metrics = metric_set(accuracy, roc_auc))
CV_nn_results %>% collect_metrics() %>% filter(.metric=="roc_auc") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

# Find Best Tuning Parameters
bestTune_nn <- CV_nn_results %>%
  select_best(metric="roc_auc")

#finalize workflow and fit it
final_nn_wf <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(ggg_train)

## Format the Predictions for Submission to Kaggle
pred_nn <- predict(final_nn_wf, new_data = ggg_test, type = "class") %>%
  bind_cols(., ggg_test) %>%
  rename(type = .pred_class) %>%
  select(id, type)

vroom_write(pred_nn, "GGG_preds_nn.csv", delim = ",")
## CV tune, finalize and predict here and save results22
## This takes a few min (10 on my laptop) so run it on becker if you want
# took a litle over a min. to run on batch, with 50 epochs, I got a low accuracy score of .34026
#increased epochs from 50 to 100 and decreased range from 1-20 and it took about a min. 45 seconds  and I got
# a higher accuracy rate of .64650

#11/11/24 boosting and bart

ggg_recipe <- recipe(type~., data = ggg_train) %>%
  step_mutate(type = as.factor(type), skip = TRUE) %>%
  step_mutate(color = as.factor(color)) %>% 
  step_dummy(color) %>% # Turn color to factor then dummy encode color
  step_mutate(id, features = id) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

#set workflow
boost_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(boost_model)


## CV tune, finalize and predict here and save results
## Set up grid of tuning values
boost_tunegrid <- grid_regular(tree_depth(), 
                               trees(), 
                               learn_rate(),levels = 5) 

## Set up K-fold CV
folds <- vfold_cv(ggg_train, v = 5, repeats=1)

## Run the CV
CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=boost_tunegrid,
            metrics=metric_set(roc_auc)) 

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc") 

## Finalize workflow and predict
final_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

## Predict
boost_ggg_predictions <- final_wf %>%
  predict(new_data =ggg_test, type="class")


## Format the Predictions for Submission to Kaggle
boost_ggg_kaggle_submission <- boost_ggg_predictions%>%
  rename(type=.pred_class) %>%
  bind_cols(., ggg_test) %>% #Bind predictions with test data
  select(id, type)  #keep Id, type for submission


## Write out the file
vroom_write(x=boost_ggg_kaggle_submission, file="boostGGGPreds.csv", delim=",")

#bart
ggg_recipe <- recipe(type~., data = ggg_train) %>%
  step_mutate(type = as.factor(type), skip = TRUE) %>%
  step_mutate(color = as.factor(color)) %>% 
  step_dummy(color) %>% # Turn color to factor then dummy encode color
  step_mutate(id, features = id) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

#set workflow
bart_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(bart_model)

## CV tune, finalize and predict here and save results

## Define a Model
bart_model <- parsnip::bart(
  trees = 1000,
  engine = "dbarts", 
  mode = "regression",
  prior_terminal_node_coef = .95, #Tune prior coefficient
  prior_terminal_node_expo = 2, #Tune prior exponent
  prior_outcome_range = 2)

## Run the CV
CV_results <- ggg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) 

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric="roc_auc")

## Combine into a Workflow and fit
final_workflow <- bart_wf %>% #sets up a series of steps that you can apply to any dataset
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)


## Run all the steps on test data
bart_preds <- predict(final_workflow, new_data = ggg_test)

## Format the Predictions for Submission to Kaggle
bart_kaggle_submission <- bart_preds %>%
  rename(type=.pred_class) %>%
  bind_cols(., ggg_test) %>% #Bind predictions with test data
  select(id, type)  #keep Id, type for submission

## Write out the file
vroom_write(x=bart_kaggle_submission, file="BartGGGPreds.csv", delim=",")