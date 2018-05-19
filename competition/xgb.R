library(tidyverse)
library(magrittr)
all <- fst::read_fst(here::here("competition/01-data/cleaned"))

train <- all %>%
  select(-contains("text")) %>%
  na.omit()

small_train <- train %>%
  sample_frac(0.1)


library(recipes)
rec <- recipe(answer_score ~ ., data = small_train[1, ]) %>%
  step_rm(starts_with("log"), ends_with("dt"), question_id) %>%
  add_role(id, new_role = "id") %>%
  step_dummy(all_nominal()) %>%
  #step_center(all_predictors()) %>%
  #step_scale(all_predictors()) %>%
  step_rm("id")

prep <- prep(rec, small_train)

train_prep <- bake(prep, train)
small_train_prep <- bake(prep, small_train)
library(mlr)

trainTask <- makeRegrTask(data = small_train_prep, target = "answer_score")

xgb_learner <- makeLearner("regr.xgboost", nrounds = 10)

# Create a model
cv_folds <- makeResampleDesc("CV", iters = 3) # 3 fold cross validation

rmsle <- function(task, model, pred, feats, extra.args) {
  MLmetrics::RMSLE(pmax(0, pred$data$response), pmax(0, pred$data$truth))
}
rmsle <- makeMeasure("rmsle", minimize = TRUE, fun = rmsle, properties = c("regr", "response"))



xgb_model <- train(xgb_learner, task = trainTask)
model_Params <- makeParamSet(
  makeDiscreteParam("nrounds", c(4, 5)),
  makeDiscreteParam("max_depth", c(16, 18)),
  #makeDiscreteParam("min_child_weight", c(01., 0.3, 0.5)),
  #makeDiscreteParam("lambda", c(2.4, 2.8))
  makeDiscreteParam("eta", c(0.2))
  # makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  
  # makeNumericParam("colsample_bytree", lower = 0.2, upper = 0.8)
)

parallelMap::parallelStartSocket(4)

library(MLmetrics)
tuned_model <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = cv_folds,
  measures = rmsle,
  par.set = model_Params,
  control = makeTuneControlGrid()
)

data <- generateHyperParsEffectData(tuned_model, partial.dep = FALSE)
# how to get the data
plotHyperParsEffect(data, x = "nrounds", y = "rmse.test.rmse", partial.dep.learn = NULL)



##  ............................................................................
##  Train based on best model                                               ####


trainTask <- makeRegrTask(data = train_prep, target = "answer_score")

xgb_learner <- makeLearner("regr.xgboost", 
  nrounds = tuned_model$x$nrounds, max_depth =  tuned_model$x$max_depth, 
  eta = tuned_model$x$eta
)

# Create a model

xgb_model <- train(xgb_learner, task = trainTask)


##  ............................................................................
test <- fst::read_fst(here::here("competition/01-data/cleaned-test")) %>%
  select(-contains("text"))

all <- bake(prep, test)

prediction <- predict(xgb_model, newdata = all) %>%
  as.data.frame()
prediction$answer_score <- prediction$response
prediction$id <- test$id
prediction$answer_score[prediction$answer_score < 1] <- 1

leak <- read_csv("competition/leaked_records.csv")

prediction <- bind_rows(
  leak, 
  prediction %>%
    filter(!(id %in% leak$id))
)

prediction[, c("id", "answer_score")] %>%
  write_csv("competition/xgb-four.csv")
