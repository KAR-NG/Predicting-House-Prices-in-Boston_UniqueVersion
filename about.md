### Description of this dataset

##### [ Target variable to Predict]

**house.value**	- Median value of owner-occupied homes in per $1000s.

##### [ Predictors ]

**charles.river** -	Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

**crime.rate** - Per capita crime rate by town.

**resid.zone** - Proportion of residential land zoned for lots over 25,000 sq.ft.

**indus.biz** -	Proportion of non-retail business acres per town.

**nitrogen.oxide** - Nitrogen oxides concentration (parts per 10 million).

**room** - Average number of rooms per dwelling.

**age** - Proportion of owner-occupied units built prior to 1940.

**dist.to.work** - Weighted mean of distances to five Boston employment centres.

**highway.index** -	Index of accessibility to radial highways.

**property.tax** - Full-value property-tax rate per per $10,000.

**pt.ratio** - Pupil-teacher ratio by town.

**black**	- 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

**lstat**	- lower status of the population (percent).



### About This Prediction

This prediction uses a machine learning algorithm, **Random Forest**, to make your predictions. It is the best algorithm in the case of this dataset after comparing its performance metrics with various potential algorithm candidates included:

* Lasso Regression  
* PLS  
* Decision Tree   
* K-Nearest Neightbour (KNN)  
* XGBoost  

This Random Forest had the highest R-squared value at nearly 90% with the lowest prediction error rate at 13.43%. R-squared value is an internal performance metric that indicates how well prediction power this random forest model has.

### Disclaimer

This is not the ordinary Boston dataset. Several minor adjustments had been made to the predictors' data to make it special and unique. The aim of this project and this application are for demonstration purposes only. 

### Creator

Author: Kar Ng

Date: October 2021

Github: https://github.com/KAR-NG 




