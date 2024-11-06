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

ggg_test <- read_csv('test.csv')
ggg_train <- read_csv('train.csv')
#ggg_trainNA <- read_csv('trainWithMissingValues.csv') 
# ggg_train
# ggg_trainNA

#factor type and color for na data
gggTrainNA$color = factor(gggTrainNA$color)
gggTrainNA$type = factor(gggTrainNA$type)

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
rmse_vec(ggg_train[is.na(gggTrainNA)], baked[is.na(ggg_trainNA)])
#0.1520589


#try a random forest model for this dataset

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
ggg_recipe <- recipe(type~., data = ggg_trainNA) %>% 
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