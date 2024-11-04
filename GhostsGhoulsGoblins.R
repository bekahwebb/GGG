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

gggTest <- read_csv('test.csv')
gggTrain <- read_csv('train.csv')
gggTrainNA <- read_csv('trainWithMissingValues.csv') 
gggTrain
gggTrainNA

#factor type and color for na data
gggTrainNA$color = factor(gggTrainNA$color)
gggTrainNA$type = factor(gggTrainNA$type)

#eda
ggplot(data=gggTrain, aes(x=type, y=bone_length)) +
geom_boxplot()

ggplot(data=gggTrain) + geom_mosaic(aes(x=product(color), fill=type))

#recipe
ggg_recipe <- recipe(type~., data = gggTrainNA) %>% 
  step_impute_median(all_numeric_predictors())

prep <- prep(ggg_recipe) 
baked <- bake(prep, new_data = gggTrainNA)  

#calculate the rmse
rmse_vec(gggTrain[is.na(gggTrainNA)], baked[is.na(gggTrainNA)])
#0.1520589
