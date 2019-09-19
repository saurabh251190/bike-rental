#Clean the environment
rm(list = ls())

#set working directory
setwd("C:/Users/Saurabh Gautam/Desktop/project")

#load file
bike=read.csv("C:/Users/Saurabh Gautam/Desktop/day.csv")

#dimensions and structure of data
dim(bike) #731  16
str(bike)

#load libraries
libraries=c("dplyr","ggplot2","rpart","DMwR","randomForest","corrgram")
lapply(X = libraries,require, character.only = TRUE)
rm(libraries)

#view data
View(bike)

#data preprocessing
bike$real_season =factor(x=bike$season,levels=c(1,2,3,4),labels=c("Spring","Summer","Fall","Winter"))
bike$real_yr=factor(x=bike$yr,levels=c(0,1),labels=c("2011","2012"))
bike$real_holiday=factor(x=bike$holiday,levels=c(0,1),labels=c("Working day","Holiday"))
bike$real_weather=factor(x=bike$weathersit,levels=c(1,2,3,4), 
                         labels=c("Clear/partly cloudy","Cloudy/Mist","Rain/Snow/Fog","Heavy Rain/Snow/Fog"))

bike$weathersit = as.factor(bike$weathersit)
bike$season = as.factor(bike$season)
bike$dteday = as.character(bike$dteday)
bike$mnth = as.factor(bike$mnth)
bike$weekday = as.factor(as.character(bike$weekday))
bike$workingday = as.factor(as.character(bike$workingday))
bike$yr = as.factor(bike$yr)
bike$holiday = as.factor(bike$holiday)

#check for missing values
missing_value=data.frame(apply(bike,2,function(x) sum(is.na(x))))

#no missing value
write.csv(missing_value,"C:/Users/Saurabh Gautam/Desktop/project/missingvalue.csv")

#Check the distribution of categorical Data using bar graph
bar1 = ggplot(data =bike, aes(x = real_season)) + geom_bar() + ggtitle("Count of Season")
bar2 = ggplot(data =bike, aes(x = real_weather)) + geom_bar() + ggtitle("Count of Weather")
bar3 = ggplot(data =bike, aes(x = real_holiday)) + geom_bar() + ggtitle("Count of Holiday")
bar4 = ggplot(data =bike, aes(x = workingday)) + geom_bar() + ggtitle("Count of Working day")
# ## Plotting plots together
gridExtra::grid.arrange(bar1,bar2,bar3,bar4,ncol=2)

#Check the distribution of numerical data using histogram
hist1 = ggplot(data =bike, aes(x =temp)) + ggtitle("Temperature") + geom_histogram(bins = 20)
hist2 = ggplot(data =bike, aes(x =hum)) + ggtitle("Humidity") + geom_histogram(bins = 20)
hist3 = ggplot(data =bike, aes(x =atemp)) + ggtitle("Feel Temperature") + geom_histogram(bins = 20)
hist4 = ggplot(data =bike, aes(x =windspeed)) + ggtitle("Windspeed") + geom_histogram(bins = 20)
gridExtra::grid.arrange(hist1,hist2,hist3,hist4,ncol=2)

#check distribution using scatter plot
scat1 = ggplot(data =bike, aes(x =temp, y = cnt)) + ggtitle("Temperature") + geom_point(color="red") + xlab("Temperature") + ylab("Bike Count")
scat2 = ggplot(data =bike, aes(x =hum, y = cnt)) + ggtitle("Humidity") + geom_point() + xlab("Humidity") + ylab("Bike Count")
scat3 = ggplot(data =bike, aes(x =atemp, y = cnt)) + ggtitle("Feel Temperature") + geom_point(color="red") + xlab("Feel Temperature") + ylab("Bike Count")
scat4 = ggplot(data =bike, aes(x =windspeed, y = cnt)) + ggtitle("Windspeed") + geom_point() + xlab("Windspeed") + ylab("Bike Count")
gridExtra::grid.arrange(scat1,scat2,scat3,scat4,ncol=2)

#outlier analysis
#box plot on mumeric values
cnames = colnames(bike[,c("temp","atemp","windspeed","hum")])
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = cnames[i]), data = bike)+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i])+
           ggtitle(paste("Box plot for",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
 
#there are outliers in windspeed 
#remove outliers in windspeed

val=bike$windspeed[bike$windspeed %in% boxplot.stats(bike$windspeed)$out]
bike=bike[which(!bike$windspeed %in% val),]
View(bike) #reduced to 718 observation

##feature selection
#correlation  only on numerical data

install.packages("usdm")
library(usdm)

df = bike[,c("temp","atemp","hum","windspeed")]
vifcor(df)

corrgram(df,order=F,upper.panel=panel.pie,text.panel=panel.txt,main="correlation plot")
#atemp is highly correlated with temp 

install.packages("corrplot")
M=cor(df)

library(corrplot)

install.packages("corrplot")
corrplot(M, method="circle")

#Positive correlations are displayed in blue 
corrplot(M, method="number")

#Feature scaling
#data is already normalized
#remove unwanted variables from data(only environmental and seasonal conditions)
real_bike=subset(bike,select=-c(instant,dteday,atemp,yr,mnth,holiday,weekday,workingday,casual,registered,
                           real_yr,real_holiday,real_weather,real_season))

View(real_bike)
#Applying Decision tree ML Algorithm for Regression

#train  and test data
set.seed(123)
train_index=sample(1:nrow(real_bike),0.8*nrow(real_bike))
train=real_bike[train_index,] #574 observation 
test=real_bike[-train_index,] #144 observation 

#rpart for regression
fit=rpart(cnt~.,data=train,method='anova')

#write rule into disk
write(capture.output(summary(fit)),"rules.txt")

#lets predict test data
bike_predictions=predict(fit,test[-6])

#evaluate the performnce of model

#calculate MAPE Mean Absolute Percentage Error Loss

MAPE = function(actual, pred){
  print(mean(abs((actual - pred)/actual)) * 100)
}
MAPE(test[,6],bike_predictions)
# MAPE 26.05408 %
# ACCURACY 73.94%
# MAE 1018.3953691      
# RMSE 1246.8818104      

#Alternative method
regr.eval(test[,6],bike_predictions, stats = c("mae","rmse","mape"))

#Plot a graph for actual vs predicted values
plot(test$cnt,type="l",lty=2,col="green")
lines(lr_predictions,col="blue")

#RANDOM FOREST FOR PREDICTION

#TRAIN DATA
rf_model = randomForest(cnt~., data = train,importance=TRUE, ntree = 200)

#Predict the test cases
rf_predictions = predict(rf_model, test[,-6])
MAPE(test[,6],rf_predictions)
regr.eval(test[,6],rf_predictions, stats = c("mae","rmse","mape"))
# MAPE 27.40%
# ACCURACY 72.60%
# MAE 1004.6482397
# RMSE 1142.0250063


# Train the data using linear regression

lr_model=lm(formula = cnt~., data = train)

# Check the summary of the model
summary(lr_model)

#Multiple R-squared:  0.5517,	Adjusted R-squared:  0.5453 
#F-statistic:  86.9 on 8 and 565 DF,  p-value: < 2.2e-16
#Predict the test cases

lr_predictions = predict(lr_model, test[,-6])

#Calculate MAPE
regr.eval(trues = test[,6], preds = lr_predictions, stats = c("mae","rmse","mape"))
MAPE(test[,6], lr_predictions)

#MAPE 25.28217%
#ACCURACY 74.72%

#Plot a graph for actual vs predicted values
plot(test$cnt,type="l",lty=2,col="green")
lines(lr_predictions,col="blue")


