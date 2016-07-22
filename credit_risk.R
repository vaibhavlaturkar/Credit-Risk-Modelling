

loan_data <- readRDS("G:/R/Credit_Risk_Modeling/loan_data_ch2.rds")
View(loan_data)


# Set seed of 567
set.seed(567)

# Store row numbers for training set: index_train
index_train = sample(1:nrow(loan_data), 2/3 * nrow(loan_data))

# Create training set: training_set
training_set <- loan_data[index_train, ]

# Create test set: test_set
test_set = loan_data[-index_train,]


# Build the logistic regression model
log_model_multi = glm(loan_status ~ age+ir_cat+grade+loan_amnt+annual_inc,family = "binomial",data=training_set)


# Obtain significance levels using summary()
summary(log_model_multi)

## Loan Amount varibale is not statistically significant 


## Fit a model with two variables Age and Ir_cat
log_model_small <- glm(loan_status~age+ir_cat,family = "binomial",data = training_set)
summary(log_model_small)

## We will predict the test data using log_model_small to get nitial idea of how good the model is at discriminating by looking at the range of predicted probabilities. A small range means that predictions for the test set cases do not lie far apart, and therefore the model might not be very good at discriminating good from bad customers.

predictions_all_small <- predict(log_model_small, newdata = test_set, type = "response")

# Look at the range of the object "predictions_all_small"
range(predictions_all_small)

# The range for predicted probabilities of default was rather small. As discussed, small predicted default probabilities are to be expected with low default rates, but building bigger models which basically means including more predictors can expand the range of your predictions.

log_model_full <- glm(loan_status~.,family = "binomial",data = training_set)


# Make PD-predictions for all test set elements using the the full logistic regression model
predictions_all_full = predict(log_model_full,test_set,type = "response")


# Look at the range of the object "predictions_all_small"
range(predictions_all_full)


# Make a binary predictions-vector using a cut-off of 15%
pred_cutoff_15 = ifelse(predictions_all_full > 0.15,1,0)

# Construct a confusion matrix
table_full <- table(test_set$loan_status,pred_cutoff_15)

## Accauray 
sum(diag(table_full))/nrow(test_set) ## 73.64 %

## Sensitvity
396/(396+641)  # 0.38

## Specificvity
6745/(6745+1915) # 0.77 

?glm

# Fit the logit, probit and cloglog-link logistic regression models
log_model_logit <- glm(loan_status ~ age + emp_cat + ir_cat + loan_amnt,
                       family = binomial(link = "logit"), data = training_set)
log_model_probit <- glm(loan_status ~ age + emp_cat + ir_cat + loan_amnt,                                        family = binomial(link = "probit"), data = training_set)

log_model_cloglog <- glm(loan_status ~ age + emp_cat + ir_cat + loan_amnt,family = binomial(link = "cloglog"), data = training_set) 

# Make predictions for all models using the test set
predictions_logit <- predict(log_model_logit, newdata = test_set, type = "response")
predictions_probit <- predict(log_model_probit, newdata = test_set, type = "response")
predictions_cloglog <- predict(log_model_cloglog, newdata = test_set, type = "response")

# Use a cut-off of 14% to make binary predictions-vectors
cutoff <- 0.14
class_pred_logit <- ifelse(predictions_logit > cutoff, 1, 0)
class_pred_probit <- ifelse(predictions_probit > cutoff, 1, 0)
class_pred_cloglog <- ifelse(predictions_cloglog > cutoff, 1, 0)

# Make a confusion matrix for the three models
tab_class_logit <- table(test_set$loan_status,class_pred_logit)
tab_class_probit <- table(test_set$loan_status,class_pred_probit)
tab_class_cloglog <- table(test_set$loan_status,class_pred_cloglog)

# Compute the classification accuracy for all three models
acc_logit <- sum(diag(tab_class_logit)) / nrow(test_set)          # 70.18 %
acc_probit <- sum(diag(tab_class_probit)) / nrow(test_set)        # 70.14 %
acc_cloglog <- sum(diag(tab_class_cloglog)) / nrow(test_set)      # 70.25 %



################################### Decision Tree #######################################

library(rpart)


tree_undersample <- rpart(loan_status ~ ., method = "class",data =training_set,control = rpart.control(cp = 0.0001))

plot(tree_undersample, uniform = TRUE)
text(tree_undersample)

#min xerror
printcp(tree_undersample)
which.min(tree_undersample$cptable[,"xerror"])
#0.000068493

# Prune the tree 
ptree_undersample <- prune(tree_undersample,cp=0.000068493)

# plot the tree
prp(ptree_undersample)


# Change the code below such that a tree is constructed with adjusted prior probabilities.
tree_prior <- rpart(loan_status ~ ., method = "class", 
                    data = training_set, parms = list(prior = c(0.7, 0.3)), 
                    control = rpart.control(cp = 0.001))

printcp(tree_prior)
# Plot the decision tree
plot(tree_prior, uniform = TRUE)

# Add labels to the decision tree 
text(tree_prior)


# A loss matrix penalizing 10 times more heavily for misclassified defaults.
tree_loss_matrix  <- rpart(loan_status ~ ., method = "class", data = training_set, 
                           parms = list(loss = matrix(c(0, 10, 1, 0), ncol = 2)), 
                           control = rpart.control(cp = 0.001))

# Plot the decision tree
plot(tree_loss_matrix, uniform = TRUE) 

# Add labels to the decision tree  
text(tree_loss_matrix)


# Plot the cross-validated error rate as a function of the complexity parameter
plotcp(tree_prior)

# Use printcp() to identify for which complexity parameter the cross-validated error rate is minimized
printcp(tree_prior)

# Create an index for of the row with the minimum xerror
index <- which.min(tree_prior$cptable[,"xerror"])

# Create tree_min
tree_min <- tree_prior$cptable[index, "CP"]

#  Prune the tree using tree_min
ptree_prior <- prune(tree_prior, cp = tree_min)

# Use prp() to plot the pruned tree
library(rpart.plot)

prp(ptree_prior)



# set a seed and run the code to construct the tree with the loss matrix again
set.seed(345)
tree_loss_matrix  <- rpart(loan_status ~ ., method = "class", data = training_set, 
                           parms = list(loss=matrix(c(0, 10, 1, 0), ncol = 2)),
                           control = rpart.control(cp = 0.001))

# Plot the cross-validated error rate as a function of the complexity parameter
printcp(tree_loss_matrix)
plotcp(tree_loss_matrix)



# Create an index for of the row with the minimum xerror
which.min(tree_loss_matrix$cptable[,"xerror"])


#  Prune the tree using tree_min
ptree_loss_matrix <- prune(tree_loss_matrix, cp = 0.0010000)


#Plot pruned tree
prp(ptree_loss_matrixr)



## Will will create a  vector contains weights of 1 for the non-defaults in the training set, and weights of 3 for defaults in the training sets. By specifying higher weights for default, the model will assign higher importance to classifying defaults correctly.

case_weights <- training_set
case_weights$case_weights <- ifelse(case_weights$loan_status == 1,3,1)
case_weights <- case_weights[,c(9)]


set.seed(123)

tree_weights <- rpart(loan_status~.,method = "class",data = training_set,weights = case_weights,control = rpart.control(cp=0.001))


## mininum xerror
printcp(tree_weights)

which.min(tree_weights$cptable[,"xerror"])
#cp = 0.0019026

ptree_weights <- prune(tree_weights,cp = 0.0019026)

#plot pruned tree
prp(ptree_weights)



# Make predictions for each of the pruned trees using the test set.
pred_undersample <- predict(ptree_undersample, newdata = test_set,  type = "class")
pred_prior <- predict(ptree_prior, newdata = test_set, type = "class")
pred_loss_matrix <- predict(ptree_loss_matrix, newdata = test_set, type = "class")
pred_weights <- predict(ptree_weights, newdata = test_set, type = "class")

# Construct confusion matrices using the predictions.
confmat_undersample <- table(test_set$loan_status, pred_undersample)
confmat_prior <- table(test_set$loan_status, pred_prior)
confmat_loss_matrix <- table(test_set$loan_status, pred_loss_matrix)
confmat_weights <- table(test_set$loan_status, pred_weights)

# Compute the accuracies
acc_undersample <- sum(diag(confmat_undersample)) / nrow(test_set)
acc_prior <- sum(diag(confmat_prior)) / nrow(test_set)
acc_loss_matrix <- sum(diag(confmat_loss_matrix)) / nrow(test_set)
acc_weights <- sum(diag(confmat_weights)) / nrow(test_set)





## ROC Curve

# Load the pROC-package
library(pROC)


# Construct the objects containing ROC-information
ROC_logit <- roc(test_set$loan_status, predictions_logit)
ROC_probit <- roc(test_set$loan_status, predictions_probit)
ROC_cloglog <-roc(test_set$loan_status, predictions_cloglog)
ROC_all_full <- roc(test_set$loan_status, predictions_all_full)

# Draw all ROCs on one plot
plot(ROC_logit)
lines(ROC_probit, col="blue")
lines(ROC_cloglog, col="red")
lines(ROC_all_full, col="green")

# Compute the AUCs
auc(ROC_logit)    # 0.6334
auc(ROC_probit)   # 0.6333
auc(ROC_cloglog)  # 0.6334 
auc(ROC_all_full) # 0.6512


## ROC CURVE for Tree

# Make predictions for each of the pruned trees using the test set.
predictions_undersample <- predict(ptree_undersample, newdata = test_set)[,2]
predictions_prior <- predict(ptree_prior, newdata = test_set)[,2]
predictions_loss_matrix <- predict(ptree_loss_matrix, newdata = test_set)[,2]
predictions_weights <- predict(ptree_weights, newdata = test_set)[,2]


# Construct the objects containing ROC-information
ROC_undersample <- roc(test_set$loan_status, predictions_undersample)
ROC_prior <- roc(test_set$loan_status,predictions_prior)
ROC_loss_matrix <- roc(test_set$loan_status,predictions_loss_matrix)
ROC_weights <- roc(test_set$loan_status,predictions_weights)

# Draw the ROC-curves in one plot
plot(ROC_undersample)
lines(ROC_prior,col="blue")
lines(ROC_loss_matrix,col="red")
lines(ROC_weights,col="green")


# AUC
auc(ROC_undersample) # 0.6304
auc(ROC_prior)       # 0.6016
auc(ROC_loss_matrix) # 0.6188
auc(ROC_weights)     # 0.6016


## AUC of Model of all full is more than other

## We will take only Model_all_full
## and delete one variable each time and check the AUC

colnames(training_set)

model_remove_amt <- glm(loan_status~grade+home_ownership+annual_inc+age+emp_cat+ir_cat,family = "binomial",data = training_set)
model_remove_grade <- glm(loan_status~loan_amnt+home_ownership+annual_inc+age+emp_cat+ir_cat,family = "binomial",data = training_set)
model_remove_home <- glm(loan_status~grade+loan_amnt+annual_inc+age+emp_cat+ir_cat,family = "binomial",data = training_set)
model_remove_inc <- glm(loan_status~grade+loan_amnt+home_ownership+age+emp_cat+ir_cat,family = "binomial",data = training_set)
model_remove_age <- glm(loan_status~grade+loan_amnt+home_ownership+annual_inc+emp_cat+ir_cat,family = "binomial",data = training_set)
model_remove_emp <- glm(loan_status~grade+loan_amnt+home_ownership+annual_inc+age+ir_cat,family = "binomial",data = training_set)
model_remove_ir <- glm(loan_status~grade+loan_amnt+home_ownership+annual_inc+emp_cat+emp_cat,family = "binomial",data = training_set)




predict_remove_amt <- predict(model_remove_amt,test_set,type = "response")
predict_remove_grade <- predict(model_remove_grade,test_set,type = "response")
predict_remove_home <- predict(model_remove_home,test_set,type = "response")
predict_remove_inc <- predict(model_remove_inc,test_set,type = "response")
predict_remove_age <- predict(model_remove_age,test_set,type = "response")
predict_remove_emp <- predict(model_remove_emp,test_set,type = "response")
predict_remove_ir <- predict(model_remove_ir,test_set,type = "response")

roc_remove_amt <- roc(test_set$loan_status,predict_remove_amt)
roc_remove_grade <- roc(test_set$loan_status,predict_remove_grade)
roc_remove_home <- roc(test_set$loan_status,predict_remove_home)
roc_remove_inc <- roc(test_set$loan_status,predict_remove_inc)
roc_remove_age <- roc(test_set$loan_status,predict_remove_age)
roc_remove_emp <- roc(test_set$loan_status,predict_remove_emp)
roc_remove_ir <- roc(test_set$loan_status,predict_remove_ir)



auc(roc_remove_amt)   # 0.6514
auc(roc_remove_grade) # 0.6438
auc(roc_remove_home)  # 0.6537
auc(roc_remove_inc)   # 0.6416
auc(roc_remove_age)   # 0.6517
auc(roc_remove_emp)   # 0.6493
auc(roc_remove_ir)    # 0.6519

summary(log_model_full)


## We will remove the insignificant variables 

log_model_sig <- glm(loan_status~grade+annual_inc+emp_cat,family = "binomial",data = training_set)

predict_sig <- predict(log_model_sig,test_set,type="response")

auc(roc(test_set$loan_status,predict_sig))
## After removing loan amt we got 0.6514 AUC
## After removing age,home and loan amount we got 0.6542 AUC
## And After removing ir_cat we got AUC as 0.65 


## PLot
plot(roc(test_set$loan_status,predict_sig))



## Acceptance rate
# Obtain the cutoff for acceptance rate 80%
cutoff <- quantile(predict_sig,0.80)


# Obtain the binary predictions.
bin_pred_prior_80 <- ifelse(predict_sig > cutoff, 1, 0)

# Obtain the actual default status for the accepted loans
# We are taking Accepted customers from our predicted data
accepted_status_prior_80 <- test_set$loan_status[bin_pred_prior_80 == 0]

# Obtain the bad rate for the accepted loans
sum(accepted_status_prior_80) / length(accepted_status_prior_80)
## 0.088 is bad rate means ,if we accept 80% of loan application there is chance of 8% of defaulters 


## We will create a  strategy table and strategy curve for the bank


strategy_bank <- function(prob_of_def){
  cutoff=rep(NA, 21)
  bad_rate=rep(NA, 21)
  accept_rate=seq(1,0,by=-0.05)
  for (i in 1:21){
    cutoff[i]=quantile(prob_of_def,accept_rate[i])
    pred_i=ifelse(prob_of_def> cutoff[i], 1, 0)
    pred_as_good=test_set$loan_status[pred_i==0]
    bad_rate[i]=sum(pred_as_good)/length(pred_as_good)}
  table=cbind(accept_rate,cutoff=round(cutoff,4),bad_rate=round(bad_rate,4))
  return(list(table=table,bad_rate=bad_rate, accept_rate=accept_rate, cutoff=cutoff))}



strategy_sig <- strategy_bank(predict_sig)

## Strategy Table
strategy_sig$table


## plot strategy table

plot(strategy_sig$accept_rate,strategy_sig$bad_rate,type = "l",xlab = "Acceptance Rate",ylab = "Bad Rate / Defaulter Rate",main = "Logistic Regression with Grade,Annual Income & Employee job Duration Variables",lwd=2)


################################ Support Vector Machines ################################

length(test_set)
