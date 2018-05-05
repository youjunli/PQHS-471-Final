library('keras')
#keras::install_keras('conda')
library(tidyverse)
library(dummies)
trn=read.table('ticdata2000.txt', header = F)
tst=read.table('ticeval2000.txt', header = F)
anyNA(trn)
anyNA(tst)
vnm=read.table('varnames.txt', header = F)
colnames(trn)=vnm$V1
#trn[1:5, 1:10]
colnames(tst)=vnm$V1[1:85]
tst.y=read.table('tictgts2000.txt', header = F)
colnames(tst.y)=vnm$V1[86]

#check which variables are numeric
mean(sapply(trn, is.numeric))
mean(sapply(tst, is.numeric))


#check outcome for train and test
1-mean(trn1$CARAVAN)      #train
1-mean(tst.y$CARAVAN)    #test



##################################################################################
#Neural Network for testing one-hot vs numerical, not included in the pdf file
#convert categorical variables to factors
trn1=trn
tst1=tst
names(trn1[,-c(2,3,86)])
trn1[,-c(2,3,86)]=as.data.frame(apply(trn1[,-c(2,3,86)],2, function(x) as.factor(x)))
sum(sapply(trn1, is.numeric))
class(trn1$MOSTYPE)
tst1[,-c(2,3)]=as.data.frame(apply(tst1[,-c(2,3)],2, function(x) as.factor(x)))
sum(sapply(tst1, is.numeric))

#merge data for one hot
trn1$grp=rep(100, nrow(trn1))
tst1$grp=rep(101, nrow(tst1))

df=rbind(trn1[,-86], tst1)
names(df)
sum(sapply(df,is.numeric))

#one hot
df_dum=dummy.data.frame(df)
colnames(df_dum[,637:638])

xtrn=dplyr::filter(df_dum, grp==100)
x_train=data.matrix(xtrn[,-ncol(xtrn)])
dim(x_train)
#sum(sapply(x_train, is.numeric))
y_train=data.matrix((trn[,86]))
xtst=filter(df_dum, grp==101)
x_test=data.matrix(xtst[,-ncol(xtst)])
dim(x_test)
y_test=data.matrix(tst.y)

use_session_with_seed(621)
model = keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")   #output
summary(model)
model %>% compile(
  loss = "binary_crossentropy",
  #optimizer = optimizer_sgd(lr=3, momentum=0.5, decay=0, nesterov=T),
  #optimizer = optimizer_rmsprop(),
  #optimizer = optimizer_adadelta(),
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
history = model %>% fit(
  x_train, y_train,
  epochs = 50, batch_size = 25, verbose = 1,
  #validation_data = list(x_test, y_test10)
  validation_split = 0.3
)



model %>% evaluate(x_test, y_test, verbose = 0) ## 0.9771 on test set
y_pred = model %>% predict_classes(x_test) ## prediction on test set
table(y_test, y_pred) ## test set confusion matrix
y_predprob = model %>% predict_proba(x_test) ## softmax probabilities, 10000 x 10 matrix
all.equal(apply(y_predprob, 1, which.max)-1, as.numeric(y_pred)) ## True


#not create dummy variables
x_train1=data.matrix((trn[,-86]))
#sum(sapply(x_train, is.numeric))
y_train1=data.matrix((trn[,86]))
x_test1=data.matrix((tst))
y_test1=data.matrix(tst.y)
dim(x_train1)
dim(x_test1)

use_session_with_seed(621)
model = keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = ncol(x_train1)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")   #output
summary(model)
model %>% compile(
  loss = "binary_crossentropy",
  #optimizer = optimizer_sgd(lr=3, momentum=0.5, decay=0, nesterov=T),
  #optimizer = optimizer_rmsprop(),
  #optimizer = optimizer_adadelta(),
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
history = model %>% fit(
  x_train1, y_train1,
  epochs = 50, batch_size = 25, verbose = 1,
  #validation_data = list(x_test, y_test10)
  validation_split = 0.3
)

model %>% evaluate(x_test1, y_test1, verbose = 0) ## 0.9771 on test set
y_pred1 = model %>% predict_classes(x_test1) ## prediction on test set
table(y_test1, y_pred1) ## test set confusion matrix
y_predprob = model %>% predict_proba(x_test) ## softmax probabilities, 10000 x 10 matrix
all.equal(apply(y_predprob, 1, which.max)-1, as.numeric(y_pred)) ## True



use_session_with_seed(621)
model = keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = ncol(x_train1)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")   #output
summary(model)
model %>% compile(
  loss = "binary_crossentropy",
  #optimizer = optimizer_sgd(lr=3, momentum=0.5, decay=0, nesterov=T),
  #optimizer = optimizer_rmsprop(),
  #optimizer = optimizer_adadelta(),
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
history = model %>% fit(
  x_train1, y_train1,
  epochs = 3, batch_size = 25, verbose = 1,
  #validation_data = list(x_test, y_test10)
  validation_split = 0.3
)






##########################################################################
#random forest
library(randomForest)
library(caret)
trn2=trn
tst2=tst
#convert level variables to ordered factors
trn2[,-c(1,2,3,5,86)]=(lapply(trn2[,-c(1,2,3,5,86)], function(x) factor(x, levels=as.character(sort(unique(x))), ordered = T)))
tst2[,-c(1,2,3,5)]=(lapply(tst2[,-c(1,2,3,5)], function(x) factor(x, levels=as.character(sort(unique(x))), ordered = T)))

#convert the two customer type variables to factors
trn2[,c(1,5)]=lapply(trn2[,c(1,5)], function(x) as.factor(x))
tst2[,c(1,5)]=lapply(tst2[,c(1,5)], function(x) as.factor(x))
#check to see if everything is ordered correctly
lapply(trn2[,-c(2,3,86)], head)


# ptm <- proc.time()
# set.seed(621)
# fit.rf=tuneRF(trn2[,-86], as.factor(trn2$CARAVAN),  ntreeTry = 1000, stepFactor = 1.5, doBest = T)
# proc.time() - ptm

tunegrid=expand.grid(.mtry=c(1:20))
cntrl=trainControl(method="repeatedcv",number=10, repeats = 5, search="grid")
ptm <- proc.time()
set.seed(621)
fit.rf=train(x=trn2[,-86],y=as.factor(trn2$CARAVAN), method="rf", tuneGrid=tunegrid, trControl=cntrl,tuneLength=10)
proc.time() - ptm

print(fit.rf)
plot(fit.rf)
rftest=as.data.frame(cbind(tst2,as.factor(tst.y$CARAVAN)))
yhat.rf=predict(fit.rf, newdata = rftest)
yguess=as.factor(c(rep(0,4000),1))[1:4000]
confusionMatrix(yguess, rftest$`as.factor(tst.y$CARAVAN)`)
confusionMatrix(data = yhat.rf, rftest$`as.factor(tst.y$CARAVAN)`)
varImp(fit.rf, scale=FALSE)





#########################################################################################
#xgboost
library(xgboost)
trn3=trn
tst3=tst

trn3$grp=rep(100, nrow(trn3))
tst3$grp=rep(101, nrow(tst3))

df3=rbind(trn3[,-86], tst3)
names(df3)
sum(sapply(df3,is.numeric))

df3[,c(1,5)]=lapply(df3[,c(1,5)], as.factor)
#one hot
df_dum3=dummy.data.frame(df3)
colnames(df_dum3)

xtrn3=dplyr::filter(df_dum3, grp==100)
mean(sapply(xtrn3, is.numeric))
xtrn3=as.data.frame(apply(xtrn3, 2, as.numeric))
x_train3=data.matrix(xtrn3[,-ncol(xtrn3)])
dim(x_train3)
#sum(sapply(x_train, is.numeric))
y_train3=data.matrix((trn3$CARAVAN))
colnames(y_train3)=colnames(tst.y)
xtst3=filter(df_dum3, grp==101)
xtst3=as.data.frame(apply(xtst3, 2, as.numeric))
x_test3=data.matrix(xtst3[,-ncol(xtst3)])
dim(x_test3)
y_test3=data.matrix(tst.y)


dtrain=xgb.DMatrix(data = x_train3, label = as.numeric(y_train3))
dtest=xgb.DMatrix(data = x_test3, label = as.numeric(y_test3))

watchlist=list(train=dtrain, test=dtest)
xgb_params=list(objective = "binary:logistic")
set.seed(621)
bst=xgb.train(params = xgb_params, data=dtrain,  max.depth=2, eta=1, nthread=1, nround=10, watchlist=watchlist,eval.metric = "error", eval.metric = "logloss")
predxg=predict(bst, dtest)
predxg=as.numeric(predxg>0.5)
confusionMatrix(predxg, tst.y$CARAVAN)
unique(predxg)

importance_matrix = xgb.importance(feature_names = nm, model = bst)
head(importance_matrix)





########################################################################################
#svm
library(e1071)
svm_train=xtrn3
svm_train$grp=as.factor(y_train3)
class(svm_train$grp)
dim(svm_train)
colnames(svm_train)[134]='CARAVAN'
svm_test=xtst3
svm_test$grp=as.factor(y_test3)
dim(svm_test)
class(svm_test$grp)
colnames(svm_test)[134]='CARAVAN'

#grid.c=expand.grid(C = seq(0.01,10,length.out = 20))
grid.c=expand.grid(C = seq(0.01,10,length.out = 10))
trctrl.svm=trainControl(method = "cv", number = 5)

#preProcess = c("center", "scale"),
ptm <- proc.time()
set.seed(621)
svm_Linear=train(CARAVAN ~., data = svm_train, method = "svmLinear",
                    trControl=trctrl.svm,
                    
                    tuneGrid = grid.c,
                    tuneLength = 10)
proc.time() - ptm
svm_Linear

predsvm=predict(svm_Linear, svm_test)
confusionMatrix(predsvm, tst.y$CARAVAN)
unique(predsvm)



##############################################################################################################
#NN
use_session_with_seed(621)
model = keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = ncol(x_train3), kernel_regularizer = regularizer_l2(l = 0.01)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 267, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")   #output
summary(model)
model %>% compile(
  loss = "binary_crossentropy",
  #optimizer = optimizer_sgd(lr=3, momentum=0.5, decay=0, nesterov=T),
  #optimizer = optimizer_rmsprop(),
  #optimizer = optimizer_adadelta(),
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)
history = model %>% fit(
  x_train3, y_train3,
  epochs = 50, batch_size = 100, verbose = 1,
  #validation_data = list(x_test, y_test10)
  validation_split = 0.3
)

# history = model %>% fit(
#   x_train3, y_train3,
#   epochs = 5, batch_size = 100, verbose = 1,
#   validation_split = 0.3
# )

model %>% evaluate(x_test3, y_test3, verbose = 0) ## 0.9771 on test set
y_pred1 = model %>% predict_classes(x_test3) ## prediction on test set
table(y_test3, y_pred1) ## test set confusion matrix

y_predprob = model %>% predict_proba(x_test) ## softmax probabilities, 10000 x 10 matrix
all.equal(apply(y_predprob, 1, which.max)-1, as.numeric(y_pred)) ## True




###############################################################
#kmeans
trn5=trn[,c(6:41)]
#determine how many clusters by within group sum of squared error
wss=(nrow(trn5)-1)*sum(apply(trn5, 2, var))
for (i in 2:40)
{
  wss[i]=sum(kmeans(trn5, centers=i, nstart = 10)$withinss)
}
plot(1:40, wss, type = 'b', xlab="Number of Clusters", ylab="Within SSE")

#choose 10
cluster.km=kmeans(trn5, 10, nstart = 10)
summary(cluster.km)

##########################################################################
#Hierarchical clustering
dist5=dist(trn5)
plot(hclust(dist5, method = 'average'), labels=F, main='Average')
plot(hclust(dist5, method = 'complete'), labels=F, main='Complete')

cluster.hc.c=cutree(hclust(dist5,method = 'complete'),10)
cluster.hc.a=cutree(hclust(dist5,method = 'average'),10)
table(cluster.hc.c, cluster.hc.a)
table(cluster.hc.c, cluster.km$cluster)
table(cluster.hc.a, cluster.km$cluster)
###########################################################################
#MDS
cmds3=cmdscale(dist5, k=3, add=T, list. = T)
x=cmds3$points[,1]
y=cmds3$points[,2]
z=cmds3$points[,3]
library(rgl)
plot3d(x, y, z, size=10, col=as.numeric(factor(colnames(trn5) )))
text3d(x, y, z, nci.labs2, cex=.7)
