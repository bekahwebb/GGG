#11/11/24 boosting and bart
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
boost_tunegrid <- grid_regular(tree_depth(range = c(2, 10)), 
trees(range = c(50, 500)), 
learn_rate(range = c(0.01, 0.3)), 
levels = 5
)

## Set up K-fold CV
folds <- vfold_cv(ggg_train, v = 5, repeats=1)

## Run the CV
CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=boost_tunegrid,
            metrics=metric_set(roc_auc, accuracy)) 

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
#start time 1:08 p end time 1:09 p
#public score .69943
#try increasing the tree range to 1000 same score :(

# #bart
# ggg_recipe <- recipe(type~., data = ggg_train) %>%
#   step_mutate(type = as.factor(type), skip = TRUE) %>%
#   step_mutate(color = as.factor(color)) %>% 
#   step_dummy(color) %>% # Turn color to factor then dummy encode color
#   step_mutate(id, features = id) %>%
#   step_range(all_numeric_predictors(), min = 0, max = 1)
# 
# bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
#   set_engine("dbarts") %>% # might need to install
#   set_mode("classification")
# 
# #set workflow
# bart_wf <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(bart_model)
# 
# ## CV tune, finalize and predict here and save results
# 
# ## Define a Model
# bart_model <- parsnip::bart(
#   trees = 1000,
#   engine = "dbarts", 
#   mode = "regression",
#   prior_terminal_node_coef = .95, #Tune prior coefficient
#   prior_terminal_node_expo = 2, #Tune prior exponent
#   prior_outcome_range = 2)
# 
# ## Run the CV
# CV_results <- bart_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) 
# 
# ## Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ## Combine into a Workflow and fit
# final_workflow <- bart_wf %>% #sets up a series of steps that you can apply to any dataset
#   finalize_workflow(bestTune) %>%
#   fit(data=ggg_train)
# 
# 
# ## Run all the steps on test data
# bart_preds <- predict(final_workflow, new_data = ggg_test)
# 
# ## Format the Predictions for Submission to Kaggle
# bart_kaggle_submission <- bart_preds %>%
#   rename(type=.pred_class) %>%
#   bind_cols(., ggg_test) %>% #Bind predictions with test data
#   select(id, type)  #keep Id, type for submission
# 
# ## Write out the file
# vroom_write(x=bart_kaggle_submission, file="BartGGGPreds.csv", delim=",")