Boston House Prices \[Unique Version\]- Regression Analysis with Machine
Learning
================
Kar Ng
2021

-   [1 R PACKAGES](#1-r-packages)
-   [2 INTRODUCTION](#2-introduction)
-   [3 DATA PREPARATION](#3-data-preparation)
    -   [3.1 Data Import](#31-data-import)
    -   [3.2 Data Description](#32-data-description)
    -   [3.3 Data Exploration](#33-data-exploration)
-   [4 DATA CLEANING](#4-data-cleaning)
    -   [4.1 Column removal and
        factorise](#41-column-removal-and-factorise)
    -   [4.2 NA Imputation](#42-na-imputation)
-   [5 EXPLORATORY DATA ANALYSIS
    (EDA)](#5-exploratory-data-analysis-eda)
    -   [5.1 Distribution Study](#51-distribution-study)
    -   [5.2 Outliers Detection](#52-outliers-detection)
    -   [5.3 Relationships](#53-relationships)
    -   [5.4 Multicollinearity](#54-multicollinearity)
-   [5 Model Building](#5-model-building)
    -   [5.1 Train-test split](#51-train-test-split)
    -   [5.2 Multiple Linear Regression
        (MLR)](#52-multiple-linear-regression-mlr)
    -   [5.3 Assumption tests of Multiple Linear
        Regressions](#53-assumption-tests-of-multiple-linear-regressions)
    -   [5.4 Lasso](#54-lasso)
    -   [5.5 PLS](#55-pls)
    -   [5.6 KNN](#56-knn)
    -   [5.7 Decision Tree / CART](#57-decision-tree--cart)
    -   [5.8 Random Forest](#58-random-forest)
    -   [5.9 Stochastic gradient boosting (*XGBoost*
        in R)](#59-stochastic-gradient-boosting-xgboost-in-r)
    -   [6.0 Final Model Comparison](#60-final-model-comparison)
-   [6 Model for Production](#6-model-for-production)
-   [7 Conclusion](#7-conclusion)
-   [8 LEGALITY](#8-legality)
-   [9 REFERENCE](#9-reference)

------------------------------------------------------------------------

![](https://raw.githubusercontent.com/KAR-NG/Predicting-House-Prices-in-Boston_UniqueVersion/main/pic1_thumbnail.png)
(*Picture by King of Hearts*)

------------------------------------------------------------------------

## 1 R PACKAGES

``` r
# R Libraries

library(tidyverse)
library(skimr)
library(caret)
library(MASS)
library(kableExtra)
library(qqplotr)
library(glmnet)
library(car)
library(corrplot)
library(mgcv)
library(randomForest)
library(doParallel)
library(pls)
library(tidytext)

# R setting

options(scipen = 0)
```

## 2 INTRODUCTION

This project uses a public dataset named “Boston” from the R package -
“MASS”. It is a famous dataset for machine learning. However, I have
made adjustment to this dataset to make overall analysis interesting and
unique. First, I changed the names of most of the columns into names
that are more representative; Second, I induced some missing values in 2
of the 14 columns, “black” and “lstat”, and I will use R to impute the
missing values with algorithm.

This Boston housing dataset studies the effects of a range of variables
on median house prices in Boston in **late 70s**, United States.

I will statistically analyse the dataset and make predictions with
machine learning algorithms. Then, I will study the effects of each
variables on the median house prices, pick a model that has the highest
predictive power, and build an online interactive application by using
RShiny in *section 6 - Model for Production*.

*Highlights of some upcoming graphs*

![](https://raw.githubusercontent.com/KAR-NG/Predicting-House-Prices-in-Boston_UniqueVersion/main/pic2_highlights.png)

## 3 DATA PREPARATION

### 3.1 Data Import

This section imports the edited version of the dataset. Following is 10
rows of data randomly selected from the dataset.

``` r
boston <- read.csv("boston.csv")
sample_n(boston, 10)
```

    ##      X crime.rate resid.zone indus.biz charles.river nitrogen.oxide  room  age
    ## 1   40    0.02763         75      2.95             0          0.428 6.595 21.8
    ## 2  126    0.16902          0     25.65             0          0.581 5.986 88.4
    ## 3  363    3.67822          0     18.10             0          0.770 5.362 96.2
    ## 4   25    0.75026          0      8.14             0          0.538 5.924 94.1
    ## 5  322    0.18159          0      7.38             0          0.493 6.376 54.3
    ## 6  286    0.01096         55      2.25             0          0.389 6.453 31.9
    ## 7  120    0.14476          0     10.01             0          0.547 5.731 65.2
    ## 8  294    0.08265          0     13.92             0          0.437 6.127 18.4
    ## 9  149    2.33099          0     19.58             0          0.871 5.186 93.8
    ## 10 442    9.72418          0     18.10             0          0.740 6.406 97.2
    ##    dist.to.work highway.index property.tax pt.ratio  black lstat house.value
    ## 1        5.4011             3          252     18.3 395.63  4.32        30.8
    ## 2        1.9929             2          188     19.1 385.02 14.81        21.4
    ## 3        2.1036            24          666     20.2 380.79 10.19        20.8
    ## 4        4.3996             4          307     21.0 394.33 16.30        15.6
    ## 5        4.5404             5          287     19.6 396.90  6.87        23.1
    ## 6        7.3073             1          300     15.3 394.72  8.23        22.0
    ## 7        2.7592             6          432     17.8 391.50 13.61        19.3
    ## 8        5.5027             4          289     16.0 396.90  8.58        23.9
    ## 9        1.5296             5          403     14.7 356.99 28.32        17.8
    ## 10       2.0651            24          666     20.2 385.96 19.52        17.1

### 3.2 Data Description

The dataset has important information regarding factors that may affect
the price of a house in Boston in late 70s. For examples, crime rate,
number of rooms in the house, nitrogen oxides concentration, its
proximity to industrial area, employment centers, highways and etc.

The unit of the house value is “median house price”. Believing this
median values is extracted from multiple houses in the respective
regions in the city.

Following is the data description adapted from the relevant R package.

``` r
Variables <- names(boston[2:15])
Description <- c("Per capita crime rate by town.",
                 "Proportion of residential land zoned for lots over 25,000 sq.ft.",
                 "Proportion of non-retail business acres per town.",
                 "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).",
                 "Nitrogen oxides concentration (parts per 10 million).",
                 "Average number of rooms per dwelling.",
                 "Proportion of owner-occupied units built prior to 1940.",
                 "Weighted mean of distances to five Boston employment centres.",
                 "Index of accessibility to radial highways.",
                 "Full-value property-tax rate per per $10,000.",
                 "Pupil-teacher ratio by town.",
                 "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.",
                 "lower status of the population (percent).",
                 "Median value of owner-occupied homes in per $1000s.")

data.frame(Variables, Description) %>% 
  kbl() %>% 
  kable_styling(bootstrap_options = c("hover", "bordered", "stripped"))
```

<table class="table table-hover table-bordered" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Variables
</th>
<th style="text-align:left;">
Description
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
crime.rate
</td>
<td style="text-align:left;">
Per capita crime rate by town.
</td>
</tr>
<tr>
<td style="text-align:left;">
resid.zone
</td>
<td style="text-align:left;">
Proportion of residential land zoned for lots over 25,000 sq.ft.
</td>
</tr>
<tr>
<td style="text-align:left;">
indus.biz
</td>
<td style="text-align:left;">
Proportion of non-retail business acres per town.
</td>
</tr>
<tr>
<td style="text-align:left;">
charles.river
</td>
<td style="text-align:left;">
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
</td>
</tr>
<tr>
<td style="text-align:left;">
nitrogen.oxide
</td>
<td style="text-align:left;">
Nitrogen oxides concentration (parts per 10 million).
</td>
</tr>
<tr>
<td style="text-align:left;">
room
</td>
<td style="text-align:left;">
Average number of rooms per dwelling.
</td>
</tr>
<tr>
<td style="text-align:left;">
age
</td>
<td style="text-align:left;">
Proportion of owner-occupied units built prior to 1940.
</td>
</tr>
<tr>
<td style="text-align:left;">
dist.to.work
</td>
<td style="text-align:left;">
Weighted mean of distances to five Boston employment centres.
</td>
</tr>
<tr>
<td style="text-align:left;">
highway.index
</td>
<td style="text-align:left;">
Index of accessibility to radial highways.
</td>
</tr>
<tr>
<td style="text-align:left;">
property.tax
</td>
<td style="text-align:left;">
Full-value property-tax rate per per $10,000.
</td>
</tr>
<tr>
<td style="text-align:left;">
pt.ratio
</td>
<td style="text-align:left;">
Pupil-teacher ratio by town.
</td>
</tr>
<tr>
<td style="text-align:left;">
black
</td>
<td style="text-align:left;">
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
</td>
</tr>
<tr>
<td style="text-align:left;">
lstat
</td>
<td style="text-align:left;">
lower status of the population (percent).
</td>
</tr>
<tr>
<td style="text-align:left;">
house.value
</td>
<td style="text-align:left;">
Median value of owner-occupied homes in per $1000s.
</td>
</tr>
</tbody>
</table>

### 3.3 Data Exploration

The dataset has 506 rows of observations and 14 columns of variables.
All variables are in numerical form.

``` r
glimpse(boston) 
```

    ## Rows: 506
    ## Columns: 15
    ## $ X              <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ~
    ## $ crime.rate     <dbl> 0.00632, 0.02731, 0.02729, 0.03237, 0.06905, 0.02985, 0~
    ## $ resid.zone     <dbl> 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.5, 12.5, 12.5, 12.5, ~
    ## $ indus.biz      <dbl> 2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7~
    ## $ charles.river  <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0~
    ## $ nitrogen.oxide <dbl> 0.538, 0.469, 0.469, 0.458, 0.458, 0.458, 0.524, 0.524,~
    ## $ room           <dbl> 6.575, 6.421, 7.185, 6.998, 7.147, 6.430, 6.012, 6.172,~
    ## $ age            <dbl> 65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 66.6, 96.1, 100.0, ~
    ## $ dist.to.work   <dbl> 4.0900, 4.9671, 4.9671, 6.0622, 6.0622, 6.0622, 5.5605,~
    ## $ highway.index  <int> 1, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4~
    ## $ property.tax   <int> 296, 242, 242, 222, 222, 222, 311, 311, 311, 311, 311, ~
    ## $ pt.ratio       <dbl> 15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 1~
    ## $ black          <dbl> 396.90, 396.90, 392.83, 394.63, 396.90, NA, 395.60, 396~
    ## $ lstat          <dbl> 4.98, 9.14, 4.03, 2.94, 5.33, 5.21, 12.43, 19.15, 29.93~
    ## $ house.value    <dbl> 24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 1~

From observation, “charles.river” is either 0 or 1, and therefore it
should be a categorical variable and it should be assigned with factor
type. The first variable, “x”, has to be removed because it is just row
number and irrelevant.

There are 12 and 17 missing data in the “black” and “lstat” column.
Fortunately, the completeness of these two variables are 97.6% and
96.6%. It is a matter of choice whether one wants to remove these values
or apply imputation techniques. I will go with the later as it is my
purpose to induce these missing values using imputation model using R.

``` r
skim_without_charts(boston)
```

<table style="width: auto;" class="table table-condensed">
<caption>
Data summary
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:left;">
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Name
</td>
<td style="text-align:left;">
boston
</td>
</tr>
<tr>
<td style="text-align:left;">
Number of rows
</td>
<td style="text-align:left;">
506
</td>
</tr>
<tr>
<td style="text-align:left;">
Number of columns
</td>
<td style="text-align:left;">
15
</td>
</tr>
<tr>
<td style="text-align:left;">
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
</td>
<td style="text-align:left;">
</td>
</tr>
<tr>
<td style="text-align:left;">
Column type frequency:
</td>
<td style="text-align:left;">
</td>
</tr>
<tr>
<td style="text-align:left;">
numeric
</td>
<td style="text-align:left;">
15
</td>
</tr>
<tr>
<td style="text-align:left;">
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
</td>
<td style="text-align:left;">
</td>
</tr>
<tr>
<td style="text-align:left;">
Group variables
</td>
<td style="text-align:left;">
None
</td>
</tr>
</tbody>
</table>

**Variable type: numeric**

<table>
<thead>
<tr>
<th style="text-align:left;">
skim\_variable
</th>
<th style="text-align:right;">
n\_missing
</th>
<th style="text-align:right;">
complete\_rate
</th>
<th style="text-align:right;">
mean
</th>
<th style="text-align:right;">
sd
</th>
<th style="text-align:right;">
p0
</th>
<th style="text-align:right;">
p25
</th>
<th style="text-align:right;">
p50
</th>
<th style="text-align:right;">
p75
</th>
<th style="text-align:right;">
p100
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
X
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
253.50
</td>
<td style="text-align:right;">
146.21
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
127.25
</td>
<td style="text-align:right;">
253.50
</td>
<td style="text-align:right;">
379.75
</td>
<td style="text-align:right;">
506.00
</td>
</tr>
<tr>
<td style="text-align:left;">
crime.rate
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
3.61
</td>
<td style="text-align:right;">
8.60
</td>
<td style="text-align:right;">
0.01
</td>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
0.26
</td>
<td style="text-align:right;">
3.68
</td>
<td style="text-align:right;">
88.98
</td>
</tr>
<tr>
<td style="text-align:left;">
resid.zone
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
11.36
</td>
<td style="text-align:right;">
23.32
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
12.50
</td>
<td style="text-align:right;">
100.00
</td>
</tr>
<tr>
<td style="text-align:left;">
indus.biz
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
11.14
</td>
<td style="text-align:right;">
6.86
</td>
<td style="text-align:right;">
0.46
</td>
<td style="text-align:right;">
5.19
</td>
<td style="text-align:right;">
9.69
</td>
<td style="text-align:right;">
18.10
</td>
<td style="text-align:right;">
27.74
</td>
</tr>
<tr>
<td style="text-align:left;">
charles.river
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
0.07
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
1.00
</td>
</tr>
<tr>
<td style="text-align:left;">
nitrogen.oxide
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
0.12
</td>
<td style="text-align:right;">
0.38
</td>
<td style="text-align:right;">
0.45
</td>
<td style="text-align:right;">
0.54
</td>
<td style="text-align:right;">
0.62
</td>
<td style="text-align:right;">
0.87
</td>
</tr>
<tr>
<td style="text-align:left;">
room
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
6.28
</td>
<td style="text-align:right;">
0.70
</td>
<td style="text-align:right;">
3.56
</td>
<td style="text-align:right;">
5.89
</td>
<td style="text-align:right;">
6.21
</td>
<td style="text-align:right;">
6.62
</td>
<td style="text-align:right;">
8.78
</td>
</tr>
<tr>
<td style="text-align:left;">
age
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
68.57
</td>
<td style="text-align:right;">
28.15
</td>
<td style="text-align:right;">
2.90
</td>
<td style="text-align:right;">
45.02
</td>
<td style="text-align:right;">
77.50
</td>
<td style="text-align:right;">
94.07
</td>
<td style="text-align:right;">
100.00
</td>
</tr>
<tr>
<td style="text-align:left;">
dist.to.work
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
3.80
</td>
<td style="text-align:right;">
2.11
</td>
<td style="text-align:right;">
1.13
</td>
<td style="text-align:right;">
2.10
</td>
<td style="text-align:right;">
3.21
</td>
<td style="text-align:right;">
5.19
</td>
<td style="text-align:right;">
12.13
</td>
</tr>
<tr>
<td style="text-align:left;">
highway.index
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
9.55
</td>
<td style="text-align:right;">
8.71
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
4.00
</td>
<td style="text-align:right;">
5.00
</td>
<td style="text-align:right;">
24.00
</td>
<td style="text-align:right;">
24.00
</td>
</tr>
<tr>
<td style="text-align:left;">
property.tax
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
408.24
</td>
<td style="text-align:right;">
168.54
</td>
<td style="text-align:right;">
187.00
</td>
<td style="text-align:right;">
279.00
</td>
<td style="text-align:right;">
330.00
</td>
<td style="text-align:right;">
666.00
</td>
<td style="text-align:right;">
711.00
</td>
</tr>
<tr>
<td style="text-align:left;">
pt.ratio
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
18.46
</td>
<td style="text-align:right;">
2.16
</td>
<td style="text-align:right;">
12.60
</td>
<td style="text-align:right;">
17.40
</td>
<td style="text-align:right;">
19.05
</td>
<td style="text-align:right;">
20.20
</td>
<td style="text-align:right;">
22.00
</td>
</tr>
<tr>
<td style="text-align:left;">
black
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
0.98
</td>
<td style="text-align:right;">
355.85
</td>
<td style="text-align:right;">
92.24
</td>
<td style="text-align:right;">
0.32
</td>
<td style="text-align:right;">
374.71
</td>
<td style="text-align:right;">
391.38
</td>
<td style="text-align:right;">
396.12
</td>
<td style="text-align:right;">
396.90
</td>
</tr>
<tr>
<td style="text-align:left;">
lstat
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
0.97
</td>
<td style="text-align:right;">
12.59
</td>
<td style="text-align:right;">
7.16
</td>
<td style="text-align:right;">
1.73
</td>
<td style="text-align:right;">
6.86
</td>
<td style="text-align:right;">
11.32
</td>
<td style="text-align:right;">
16.94
</td>
<td style="text-align:right;">
37.97
</td>
</tr>
<tr>
<td style="text-align:left;">
house.value
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
22.53
</td>
<td style="text-align:right;">
9.20
</td>
<td style="text-align:right;">
5.00
</td>
<td style="text-align:right;">
17.02
</td>
<td style="text-align:right;">
21.20
</td>
<td style="text-align:right;">
25.00
</td>
<td style="text-align:right;">
50.00
</td>
</tr>
</tbody>
</table>

Alternatively, I can examine missing data in the dataset by following
code. There are 12 missing data in the column “black”, and 17 from the
column “lstat”.

``` r
colSums(is.na(boston))
```

    ##              X     crime.rate     resid.zone      indus.biz  charles.river 
    ##              0              0              0              0              0 
    ## nitrogen.oxide           room            age   dist.to.work  highway.index 
    ##              0              0              0              0              0 
    ##   property.tax       pt.ratio          black          lstat    house.value 
    ##              0              0             12             17              0

Following is a summary summarises some basic descriptive statistics of
the dataset.

``` r
summary(boston)
```

    ##        X           crime.rate         resid.zone       indus.biz    
    ##  Min.   :  1.0   Min.   : 0.00632   Min.   :  0.00   Min.   : 0.46  
    ##  1st Qu.:127.2   1st Qu.: 0.08205   1st Qu.:  0.00   1st Qu.: 5.19  
    ##  Median :253.5   Median : 0.25651   Median :  0.00   Median : 9.69  
    ##  Mean   :253.5   Mean   : 3.61352   Mean   : 11.36   Mean   :11.14  
    ##  3rd Qu.:379.8   3rd Qu.: 3.67708   3rd Qu.: 12.50   3rd Qu.:18.10  
    ##  Max.   :506.0   Max.   :88.97620   Max.   :100.00   Max.   :27.74  
    ##                                                                     
    ##  charles.river     nitrogen.oxide        room            age        
    ##  Min.   :0.00000   Min.   :0.3850   Min.   :3.561   Min.   :  2.90  
    ##  1st Qu.:0.00000   1st Qu.:0.4490   1st Qu.:5.886   1st Qu.: 45.02  
    ##  Median :0.00000   Median :0.5380   Median :6.208   Median : 77.50  
    ##  Mean   :0.06917   Mean   :0.5547   Mean   :6.285   Mean   : 68.57  
    ##  3rd Qu.:0.00000   3rd Qu.:0.6240   3rd Qu.:6.623   3rd Qu.: 94.08  
    ##  Max.   :1.00000   Max.   :0.8710   Max.   :8.780   Max.   :100.00  
    ##                                                                     
    ##   dist.to.work    highway.index     property.tax      pt.ratio    
    ##  Min.   : 1.130   Min.   : 1.000   Min.   :187.0   Min.   :12.60  
    ##  1st Qu.: 2.100   1st Qu.: 4.000   1st Qu.:279.0   1st Qu.:17.40  
    ##  Median : 3.207   Median : 5.000   Median :330.0   Median :19.05  
    ##  Mean   : 3.795   Mean   : 9.549   Mean   :408.2   Mean   :18.46  
    ##  3rd Qu.: 5.188   3rd Qu.:24.000   3rd Qu.:666.0   3rd Qu.:20.20  
    ##  Max.   :12.127   Max.   :24.000   Max.   :711.0   Max.   :22.00  
    ##                                                                   
    ##      black            lstat        house.value   
    ##  Min.   :  0.32   Min.   : 1.73   Min.   : 5.00  
    ##  1st Qu.:374.71   1st Qu.: 6.86   1st Qu.:17.02  
    ##  Median :391.38   Median :11.32   Median :21.20  
    ##  Mean   :355.85   Mean   :12.59   Mean   :22.53  
    ##  3rd Qu.:396.12   3rd Qu.:16.94   3rd Qu.:25.00  
    ##  Max.   :396.90   Max.   :37.97   Max.   :50.00  
    ##  NA's   :12       NA's   :17

*Insights*

-   I can see that there are missing values in the “black” and “lstat”
    as represented by “NA”.

-   The feature “charles.river” is a binary type with either 0 or 1.

## 4 DATA CLEANING

This part converts the dataset into a format that is appropriate for
analysis or storage.

Depending on context, my usual cleaning techniques include but not
limited to the following:

-   Renaming of columns and levels if required.  
-   Long-wide structure transformation if required.  
-   Replacing NA by any appropriate imputation means.  
-   Removing variables that do not contribute or irrelevant to the
    prediction of the responding variable.  
-   Removing variables that have too many missing values (generally said
    a 60% missing values).  
-   Removing rows that have many missing values.  
-   Factorise variables (or known as features).  
-   Feature engineering if required.

Several specific cleaning tasks identified from the previous section:

-   Remove the first column “x”.  
-   Convert “charles.river” from integer into factor.  
-   Imputation and create new associated features.

### 4.1 Column removal and factorise

Following code remove first column “X” which is the row number, and
factorise the binary column, “charles.river” to convert it into factor.

``` r
boston <- boston %>% 
  dplyr::select(-1) %>% 
  mutate(charles.river = as.factor(charles.river))
```

### 4.2 NA Imputation

I am applying *caret* function for this imputation. I will first convert
the factor variable “charles river” in the dataset into dummy data as it
is required by the package *caret* imputation to work.

``` r
# Dummy transformation of factorised Charles river because caret function only work with dummy data.  

dummy.variable <-  dummyVars(~., data = boston[, -14])
boston.dummy <- dummy.variable %>% predict(boston[, -14])
```

Assessing again the number of missing values in this transformed
dataset.

``` r
colSums(is.na(boston.dummy)) 
```

    ##      crime.rate      resid.zone       indus.biz charles.river.0 charles.river.1 
    ##               0               0               0               0               0 
    ##  nitrogen.oxide            room             age    dist.to.work   highway.index 
    ##               0               0               0               0               0 
    ##    property.tax        pt.ratio           black           lstat 
    ##               0               0              12              17

Now, I am building an imputation model that uses all columns in the
dataset to predict these missing values, by the method of “Bagged
Decision Trees”.

``` r
# Imputation

imputation.model <- preProcess(boston.dummy, method = "bagImpute")
imputed.boston <- imputation.model %>% predict(boston.dummy)
```

And the missing values have now been filled up by imputation.

``` r
colSums(is.na(imputed.boston))
```

    ##      crime.rate      resid.zone       indus.biz charles.river.0 charles.river.1 
    ##               0               0               0               0               0 
    ##  nitrogen.oxide            room             age    dist.to.work   highway.index 
    ##               0               0               0               0               0 
    ##    property.tax        pt.ratio           black           lstat 
    ##               0               0               0               0

Over-writing the original “black” and “lstat” with the newly imputed
“black” and “lstat” columns.

``` r
boston$black <- imputed.boston[, 13]
boston$lstat <- imputed.boston[, 14]
```

Now, there are no more missing values in the dataset.

``` r
colSums(is.na(boston))
```

    ##     crime.rate     resid.zone      indus.biz  charles.river nitrogen.oxide 
    ##              0              0              0              0              0 
    ##           room            age   dist.to.work  highway.index   property.tax 
    ##              0              0              0              0              0 
    ##       pt.ratio          black          lstat    house.value 
    ##              0              0              0              0

Since the purpose of this project is to assess the effects of each of
these variables in predicting the median house prices. I will keep all
variables at this point. I will make variable selection again in later
analysis.

Let’s have a final glimpse of the data again.

``` r
# Convert 2 integers variable "highway.index" and "property.tax " to double type. 

boston <- boston %>% 
  mutate_if(is.integer, as.double)

glimpse(boston)
```

    ## Rows: 506
    ## Columns: 14
    ## $ crime.rate     <dbl> 0.00632, 0.02731, 0.02729, 0.03237, 0.06905, 0.02985, 0~
    ## $ resid.zone     <dbl> 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.5, 12.5, 12.5, 12.5, ~
    ## $ indus.biz      <dbl> 2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7~
    ## $ charles.river  <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0~
    ## $ nitrogen.oxide <dbl> 0.538, 0.469, 0.469, 0.458, 0.458, 0.458, 0.524, 0.524,~
    ## $ room           <dbl> 6.575, 6.421, 7.185, 6.998, 7.147, 6.430, 6.012, 6.172,~
    ## $ age            <dbl> 65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 66.6, 96.1, 100.0, ~
    ## $ dist.to.work   <dbl> 4.0900, 4.9671, 4.9671, 6.0622, 6.0622, 6.0622, 5.5605,~
    ## $ highway.index  <dbl> 1, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4~
    ## $ property.tax   <dbl> 296, 242, 242, 222, 222, 222, 311, 311, 311, 311, 311, ~
    ## $ pt.ratio       <dbl> 15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 1~
    ## $ black          <dbl> 396.9000, 396.9000, 392.8300, 394.6300, 396.9000, 387.2~
    ## $ lstat          <dbl> 4.98000, 9.14000, 4.03000, 2.94000, 5.33000, 5.21000, 1~
    ## $ house.value    <dbl> 24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 1~

Perfection.

## 5 EXPLORATORY DATA ANALYSIS (EDA)

Transforming the data frame for effective EDA.

``` r
bos <- boston %>% 
  gather(key = "key", value = "value") %>% 
  mutate(value = as.numeric(value))
```

### 5.1 Distribution Study

This section studies data distribution of each variables. In general,
any non-skewed distribution that closes to a Gaussian distribution would
be more useful for the prediction of the house value. They would have
good relation with the responding variable - “house.value”.

``` r
ggplot(bos, aes(x = value, fill = key)) +
  geom_histogram(colour = "white", bins = 20) +
  facet_wrap(~ key, scale = "free") + 
  theme(legend.position = "none")
```

![](boston_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

Insight:

-   The *charles.river* has a binary distribution with value in either 1
    or 0.  
-   The *room* has a distribution that closes to Gaussian
    distribution.  
-   The y variable *house.value* has a nearly Gaussian distributed
    distribution which is a good sign.  
-   Many variables except *charles.river* and *room* seems to have
    skewed to both directions.

This part would help in manual selection of features when trying to make
the prediction based on traditional linear regression that sensitive to
outliers. Outliers are probably the reasons causing these skews.
However, choosing a type of machine learning model that robust to
outliers, such as decision tree, will be an alternative powerful option.

### 5.2 Outliers Detection

This section uses boxplot as an alternative visualization of outliers.

``` r
ggplot(bos, aes(x = value, fill = key)) +
  geom_boxplot() +
  facet_wrap(~ key, scale = "free") +
  theme(legend.position = "none")
```

![](boston_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

Outliers exists in many variables include *black, crime.rate,
dist.to.work, lstat, house.value, ptratio, resid.zone,* and even in the
Gaussian distributed *room*。

The distribution plots and box plots show that the assumptions of linear
regression have been violated and therefore a non-linear regression
would perform better than linear regression algorithm, such as KNN and
tree-base algorithms.

### 5.3 Relationships

In relation to median house prices, visualisation shows that:

``` r
bos2 <- boston %>% 
  mutate(charles.river = as.numeric(charles.river)) %>% 
  pivot_longer(c(1:13), names_to = "variable", values_to = "result") %>% 
  arrange(variable)

# plot

ggplot(bos2, aes(x = result, y = house.value, colour = variable)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~variable, scales = "free_x", ncol = 3) +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5, vjust = 2),
        strip.text = element_text(size = 10)) +
  geom_smooth(se = F) +
  labs(x = "Variables",
       y = "Median House Price, /$1000",
       title = "The Impact of Environmental Features on Median House Prices")
```

![](boston_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

Insights:

-   The variables that related (either positive or negative) the most
    with the median house prices are “*room*”, “*lstat*”, “*pt.ratio*”,
    and probably “*resid.zone*”.

-   The relationships of other variables with the median house price are
    not as much.

-   The relationships between the median house prices and each of the
    independent variables are not linear.

### 5.4 Multicollinearity

This section tests for the existence of multicollinearity within the
dataset. It is a problem when two or more predictor variables are
correlated with each other. One of the assumption of linear regression
is to have predictor variables independent from each other.
Multicollinearity can violate this assumption.

The existence of multicollinearity will increase the standard errors of
coefficients estimates in the linear regression model, eventually alter
the 95% Confidence intervals and finally affect the accuracy of P-values
of each variables in relation to the median house prices. These P-values
are the statistical metrics that we use to evaluate the significance of
relationships between predictors with the median house prices.

A correlogram is graphed to study the interaction between each pair of
the independent variables.

``` r
boston_cor <- boston %>% 
  mutate(charles.river = as.numeric(charles.river)) %>% 
  cor()

corrplot::corrplot(boston_cor, method = "number", type = "lower")
```

![](boston_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

Generally speaking, an absolute correlation value that is around and
higher than 50% indicates a moderate correlation. The closer to 1 in
either positive or negative direction, the higher the relationship
between the two variables. As a rule thumb, multicollinearity problem is
an issue when the correlation between the two independent variables
exceed 0.8 or -0.8.

Therefore, multicollinearity issue is likely to happen between “highway
index” and “property tax”. They have a correlation degree of 0.91. To
avoid multicollinearity problem, one should avoid the coexistence of the
two variables in a model. This can be achieved by manual selection or
auto-selection by machine learning algorithms. Alternatively, a machine
learning algorithm that immune to multicollinearity should be selected
during model building.

An alternative, popular multicollinearity detection method, called
variance inflation factor (VIF) will be performed in next section. This
method requires a model to be built prior to evaluation. VIF will
further confirm the result of multicollinearity detection by this
correlogram.

## 5 Model Building

### 5.1 Train-test split

This section creates data partitions into 80% of train set and 20% test
set. The train set will be used to build models and the test set will be
used to evaluate the performance of each models.

``` r
set.seed(123)

# Create data partition

train.index <- boston$house.value %>% createDataPartition(p = 0.8, list = F)

# Get train and test set

train.data <- boston[train.index, ]
test.data <- boston[-train.index, ]
```

### 5.2 Multiple Linear Regression (MLR)

A multiple linear regression is created to study the effect of each
variables on the median house prices.

``` r
# Using original dataset to include all data into this model

model_mlr <- lm(house.value ~., data = train.data)
```

As mentioned, a variance inflation factor (VIF) evaluation is performed.
A VIF that exceeds 5 indicates a problematic amount of collinearity
(James et al. 2014). The result matches the outcome from the previous
correlogram that “highway.index” and “property.tax” correlated and
should not be coexist in a linear regression model to avoid the issue of
multicollinearity.

``` r
vif(model_mlr)
```

    ##     crime.rate     resid.zone      indus.biz  charles.river nitrogen.oxide 
    ##       1.851747       2.322910       3.954998       1.066698       4.448657 
    ##           room            age   dist.to.work  highway.index   property.tax 
    ##       1.952910       3.223066       4.063831       8.150571       9.660433 
    ##       pt.ratio          black          lstat 
    ##       1.851009       1.354782       3.168298

Following are two linear models built with the exception of either
highway.index or property.tax. Their adjusted R-Squared will be compared
and the model with the higher adjusted R-squared will be selected.

``` r
set.seed(123)

model2 <- lm(house.value ~. - highway.index , data = train.data)

summary(model2)
```

    ## 
    ## Call:
    ## lm(formula = house.value ~ . - highway.index, data = train.data)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -15.1601  -2.6479  -0.5569   1.5288  30.0735 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     2.996e+01  5.633e+00   5.319 1.75e-07 ***
    ## crime.rate     -5.498e-02  3.947e-02  -1.393 0.164355    
    ## resid.zone      3.189e-02  1.656e-02   1.925 0.054932 .  
    ## indus.biz      -8.744e-02  7.043e-02  -1.241 0.215177    
    ## charles.river1  2.781e+00  9.732e-01   2.857 0.004501 ** 
    ## nitrogen.oxide -1.494e+01  4.483e+00  -3.332 0.000945 ***
    ## room            3.921e+00  4.701e-01   8.340 1.26e-15 ***
    ## age             1.924e-04  1.613e-02   0.012 0.990492    
    ## dist.to.work   -1.423e+00  2.400e-01  -5.929 6.65e-09 ***
    ## property.tax    2.313e-03  2.727e-03   0.848 0.396935    
    ## pt.ratio       -8.568e-01  1.540e-01  -5.564 4.88e-08 ***
    ## black           1.007e-02  3.039e-03   3.313 0.001010 ** 
    ## lstat          -5.138e-01  6.256e-02  -8.213 3.13e-15 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4.998 on 394 degrees of freedom
    ## Multiple R-squared:  0.7116, Adjusted R-squared:  0.7028 
    ## F-statistic:    81 on 12 and 394 DF,  p-value: < 2.2e-16

``` r
set.seed(123)

model3 <- lm(house.value ~. - property.tax , data = train.data)

summary(model3)
```

    ## 
    ## Call:
    ## lm(formula = house.value ~ . - property.tax, data = train.data)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -15.4425  -2.9579  -0.5023   1.6788  29.1700 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     34.786848   5.761530   6.038 3.61e-09 ***
    ## crime.rate      -0.096160   0.040374  -2.382 0.017706 *  
    ## resid.zone       0.028161   0.016014   1.759 0.079423 .  
    ## indus.biz       -0.093119   0.066305  -1.404 0.160988    
    ## charles.river1   2.699898   0.959018   2.815 0.005118 ** 
    ## nitrogen.oxide -18.413445   4.451931  -4.136 4.32e-05 ***
    ## room             3.739067   0.468279   7.985 1.56e-14 ***
    ## age              0.003536   0.015979   0.221 0.824962    
    ## dist.to.work    -1.410283   0.237298  -5.943 6.16e-09 ***
    ## highway.index    0.151276   0.047886   3.159 0.001705 ** 
    ## pt.ratio        -1.000163   0.154260  -6.484 2.68e-10 ***
    ## black            0.011205   0.003009   3.723 0.000225 ***
    ## lstat           -0.517955   0.061845  -8.375 9.79e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4.941 on 394 degrees of freedom
    ## Multiple R-squared:  0.7182, Adjusted R-squared:  0.7096 
    ## F-statistic: 83.67 on 12 and 394 DF,  p-value: < 2.2e-16

The model without property.tax (model 3) has higher adjusted R-squared
at 0.708 as compared to 0.7012 of the previous model without
highway.index. Additionally, the RSE of this model (model 3) is 4.954
which is lower than the previous model (model 2) built with property tax
at 5.012. Therefore, property.tax should be dropped to avoid
multicollinearity.

**Model performance**

-   P-value of the F-Statistics of this model is &lt; 0.05, indicating
    that there is at least of the predictor variable is significantly
    related to the median house prices.

-   The adjusted R-squared of this model is 0.7091, which is a good
    value indicating that this multiple linear regression model is able
    to explain 70.91% of the variation in the median house prices.

-   The Residual standard error (RSE) is 4.954, This corresponds to an
    error rate of 22%, which is acceptable but high enough to
    investigate a better model for prediction.

``` r
4.954/mean(Boston$medv)
```

    ## [1] 0.2198572

**Insights from coefficient estimates**

Following visualisation indicates the strength of coefficients and
significance level of each variable in relation to the median house
price.

``` r
# set up df

coef_plot <- data.frame(summary(model3)$coef) %>% 
  rownames_to_column(var = "variable") %>% 
  filter(variable != "(Intercept)") %>% 
  rename(P_value = "Pr...t..") %>% 
  mutate(sig = case_when(P_value < 0.05 & P_value > 0.01 ~ "*",
                         P_value < 0.01 & P_value > 0.001 ~ "**",
                         P_value < 0.001 ~ "***",
                         TRUE ~ " "))

# plot

plot_mlr <- ggplot(coef_plot, aes(y = Estimate, x = fct_reorder(variable, -Estimate), fill = variable)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, size = 10, hjust = 0.7),
        plot.margin = unit(c(1,1,1,1), "cm")) +
  geom_text(aes(label = paste0("(", round(Estimate, 2), ")")), vjust = 1) +
  geom_text(aes(label = sig), size = 8) +
  labs(x = "Variables", 
       y = "Coefficient Estimate",
       subtitle = "*: P<0.05, **: P<0.01, ***: P<0.001",
       title = "Coefficient of Variables in Relation to Median House Prices")+
  scale_y_continuous(lim = c(-20, 5), breaks = seq(-20, 5, 5))


plot_mlr
```

![](boston_files/figure-gfm/unnamed-chunk-27-1.png)<!-- --> Insights
from this section:

-   Variables “resid.zone”, “age”, and “indus.biz” do not have
    significant relationship with the median house prices.

-   Variable “room” has the highest significant positive impact on
    median house prices.

-   Variable “nitrogen oxide” has the highest significant negative
    impact on median house prices.

### 5.3 Assumption tests of Multiple Linear Regressions

This section describes whether the assumptions of linear regression is
fulfilled.

-   The relationship between independent variables and the responding
    variable is non-linear.

``` r
plot(model3, 1)
```

![](boston_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

-   The second assumption is that independent variables should be
    independent from each other. It is not the case in this dataset as
    there is multicollinearity problem between “rad” and “tax”.

-   Following plot the standardised residual against the fitted values
    shows that the amount of error is not similar at each data point of
    the linear model and therefore it has a feature of
    heteroscedasticity.

``` r
plot(model3, 3)
```

![](boston_files/figure-gfm/unnamed-chunk-29-1.png)<!-- --> \* There is
also no multivariate normality shown by a standard Q-Q plot below formed
by the multiple linear regression model. An ideal trend would be having
all points falling near the straight line in the middle of the plot and
form a line.

``` r
plot(model3, 2)
```

![](boston_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

This dataset has a non-parametric characteristics and therefore a
non-parameter machine learning prediction model should be selected.

### 5.4 Lasso

This section I will still apply a parametric algorithm, Lasso
regression, to check out how would the L1-norm lambda regularisation of
this method performs. The performance metrics of this method will be
used as a baseline model to compared with non-parametric machine
learning algorithms that I am going to apply later.

``` r
set.seed(123)

# build the model

model_lasso <- train(house.value ~., data = boston,
                     method = "glmnet",
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10,
                                              repeats = 3),
                     tuneGrid = expand.grid(alpha = 1,
                                            lambda = 10^seq(-3, 3, length = 100)))
```

From above lasso model, I built a grid that contains many lambda value,
and it seems like the best one will be closing to 0. A best lambda value
is the one that having the lowest RMSE.

``` r
plot(model_lasso)
```

![](boston_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

R *caret* package automatically identified the best lambda for me which
is 0.0284.

``` r
model_lasso$bestTune
```

    ##    alpha     lambda
    ## 25     1 0.02848036

Interesting, many materials state that Lasso regression should be used
when there is multicollinearity issue in the dataset because Lasso will
select 1 variable from the highly correlated group. However, it is not
happening in my case. The “highway.index” and “property.tax” are
correlated with a VIF above 5 and therefore they should not coexist in a
model as it would affect the accuracy of the coefficients.

``` r
coef(model_lasso$finalModel, model_lasso$bestTune$lambda)
```

    ## 14 x 1 sparse Matrix of class "dgCMatrix"
    ##                           s1
    ## (Intercept)     3.326632e+01
    ## crime.rate     -9.716050e-02
    ## resid.zone      4.102652e-02
    ## indus.biz       .           
    ## charles.river1  2.717758e+00
    ## nitrogen.oxide -1.669319e+01
    ## room            3.975616e+00
    ## age            -4.501236e-04
    ## dist.to.work   -1.387331e+00
    ## highway.index   2.499407e-01
    ## property.tax   -9.864180e-03
    ## pt.ratio       -9.366091e-01
    ## black           1.028716e-02
    ## lstat          -4.915874e-01

It is likely that a the best lambda is determined based on the lowest
point of RMSE. Therefore, I initiate a trade-off here. I manually
selected the lambda value that will exclude either one from the high
correlated group.

``` r
# build the model

model_lasso <- train(house.value ~., data = boston,
                     method = "glmnet",
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10,
                                              repeats = 3),
                     tuneGrid = expand.grid(alpha = 1,
                                            lambda = 0.19)) # I manually found that lambda 0.19 have property.tax removed that is highly correlated with highway.index


coef(model_lasso$finalModel, model_lasso$bestTune$lambda)
```

    ## 14 x 1 sparse Matrix of class "dgCMatrix"
    ##                           s1
    ## (Intercept)     22.597158818
    ## crime.rate      -0.041784143
    ## resid.zone       0.017023150
    ## indus.biz       -0.010704601
    ## charles.river1   2.497219549
    ## nitrogen.oxide -10.119376387
    ## room             4.309445180
    ## age              .          
    ## dist.to.work    -0.849512857
    ## highway.index    0.004578149
    ## property.tax     .          
    ## pt.ratio        -0.833521615
    ## black            0.008769646
    ## lstat           -0.490496369

Following compute the R2 and RMSE of Lasso from predicting the test
data. It will be recorded in the last section for a grand model
comparison.

``` r
# predictions

prediction_lasso <- model_lasso %>% predict(test.data)

# model performance

R2_lasso <- caret::R2(prediction_lasso, test.data$house.value)
RMSE_lasso <- RMSE(prediction_lasso, test.data$house.value)

R2_lasso
```

    ## [1] 0.7653138

``` r
RMSE_lasso
```

    ## [1] 4.659806

### 5.5 PLS

A principle component technique is applied in this section, called
Partial Least Squares (PLS). PLS is a method that can avoid
multicollinearity between PLS summarises the predictors into a few new
variables called principle component. Then, these new variables are fit
to linear regression model.

``` r
set.seed(123)

model_pls <- train(house.value ~., data = train.data,
                   method = "pls",
                   scale = TRUE,
                   trControl = trainControl(method = "repeatedcv",
                                            number = 10, 
                                            repeats = 3),
                   tuneLength = 10)
```

This is a plot showing how are the numbers of principal components (PCs)
in relation to RMSE. The lower the RMSE, the lower the model. The
optimum level of PC should be around 8 to 10.

``` r
plot(model_pls)
```

![](boston_files/figure-gfm/unnamed-chunk-38-1.png)<!-- -->

R can help to identify the best number of components (ncomp) to be used
in the model, which is 9. This value will be automatically set as the
default “ncomp” to be used during prediction.

``` r
model_pls$bestTune
```

    ##   ncomp
    ## 9     9

Internally, this model captures 90.89% of variation in the 9 components
and 72.21% of the outcome variation.

``` r
summary(model_pls$finalModel)
```

    ## Data:    X dimension: 407 13 
    ##  Y dimension: 407 1
    ## Fit method: oscorespls
    ## Number of components considered: 9
    ## TRAINING: % variance explained
    ##           1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps
    ## X           46.89    57.23    64.44    69.92    75.72    79.80    83.18
    ## .outcome    48.86    69.19    70.66    71.64    72.05    72.19    72.28
    ##           8 comps  9 comps
    ## X           85.80    90.87
    ## .outcome    72.35    72.36

In predicting the test variable, PLS has done a little better job than
lasso in terms of R2 and RMSE. These metrics will be recorded for final
grand comparison.

``` r
# Predictions

predictions_pls <- model_pls %>% predict(test.data)

# Model performance

caret::R2(predictions_pls, test.data$house.value)
```

    ## [1] 0.7688283

``` r
caret::RMSE(predictions_pls, test.data$house.value)
```

    ## [1] 4.529281

### 5.6 KNN

K-Nearest Neighbors (KNN) is the very first non-parametric model to be
implemented. It does not need to comply with parametric assumptions as
well as the collinearity issue. This method uses the neighboring points
to make estimations during prediction.

``` r
model_knn <- train(house.value ~., data = boston,
                   method = "knn",
                   trControl = trainControl(method = "repeatedcv", 
                                            number = 10,
                                            repeats = 3),
                   preProcess = c("center", "scale"),
                   tuneLength = 10
                   )
```

Grabbing the nearest 5 neighboring points during internal estimation and
prediction had the lowest RMSE. Therefore, it is the optimum “k” nearest
point that this model will be using during prediction on the test
dataset.

``` r
plot(model_knn)
```

![](boston_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

``` r
model_knn$bestTune
```

    ##   k
    ## 1 5

Great, the result has an significant improvement from the parameter
options.

``` r
# Predictions

predictions_knn <- model_knn %>% predict(test.data)

# Model performance

caret::R2(predictions_knn, test.data$house.value)
```

    ## [1] 0.8837879

``` r
caret::RMSE(predictions_knn, test.data$house.value)
```

    ## [1] 3.47336

### 5.7 Decision Tree / CART

Decision tree, also known as CART (Classification and Regression Tree)
is my second non-parametric machine learning algorithm that work should
very well in condition when there is highly non-linear relationship
between the predictors and the responding variable. Decision tree has a
upside-down tree-like structure with decision rules in each of the
branch to guide the prediction of new observations.

I am expecting a better result because decision tree tend to prefer a
situation that a few of variables in the variables list are more
powerful than the others. This house dataset has this characteristic.
Few of the powerful features are nitrogen.oxide, room, and
charles.river. This method is also immune to multicollinearity.

``` r
set.seed(123)

model_DT <- train(house.value ~., data = train.data,
                  method = "rpart", 
                  trControl = trainControl(method = "repeatedcv", 
                                            number = 10,
                                            repeats = 3),
                  tuneLength = 10)   # Specifying the number of possible cp values for pruning, default is 3. 
```

Complexity parameter (CP) is the index set to prune the tree to avoid
overfitting. The best CP detected is 0.0086 with the lowest RMSE.

``` r
plot(model_DT)
```

![](boston_files/figure-gfm/unnamed-chunk-47-1.png)<!-- -->

``` r
model_DT$bestTune
```

    ##            cp
    ## 1 0.008601926

Following decision tree can help to visualise the important decisions to
be made during each split.

``` r
par(xpd = NA)
plot(model_DT$finalModel, uniform = T, branch = 0.5, compress = T)
text(model_DT$finalModel, col = "darkgreen", fancy = F)
```

![](boston_files/figure-gfm/unnamed-chunk-49-1.png)<!-- -->

Decision tree specifies the variable “room” as the root node (the node
at the very top) after trying out all variables. On the root node, it
appears that “room” leads to the purest two branches compared to other
variables. Formally, It means that the information gain for the variable
“room” is the highest. From the root node, we will know more about
median house prices after looking at “room” compared to other variables.
In regression decision tree, the value of each split cutoff point (for
example “room &lt; 6.838”) is selected to result the purest branch, or
defined that residual sum of squared error (RSS) is minimized.

This computation is iterated at each resulting branches on sub-samples
resulting from individual parent nodes until either maximum depth of the
tree or when pure branches are achieved. This process is known as
recursive partitioning.

To improve the visual, following is an alternative visual showing the
exact same information as above tree however it shows the number of
samples and associated proportion during each split.

``` r
library(rattle)
```

    ## Loading required package: bitops

    ## 
    ## Attaching package: 'bitops'

    ## The following object is masked from 'package:Matrix':
    ## 
    ##     %&%

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    ## 
    ## Attaching package: 'rattle'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     importance

``` r
fancyRpartPlot(model_DT$finalModel, palettes = "Oranges")
```

![](boston_files/figure-gfm/unnamed-chunk-50-1.png)<!-- -->

Following is the decision rules defined in the model.

``` r
model_DT$finalModel
```

    ## n= 407 
    ## 
    ## node), split, n, deviance, yval
    ##       * denotes terminal node
    ## 
    ##  1) root 407 34125.5800 22.51057  
    ##    2) room< 6.8375 334 12681.2200 19.62695  
    ##      4) lstat>=14.4 145  3819.0870 15.30483  
    ##        8) nitrogen.oxide>=0.657 68   876.7047 12.45882  
    ##         16) crime.rate>=7.16463 46   350.9915 10.85870 *
    ##         17) crime.rate< 7.16463 22   161.6695 15.80455 *
    ##        9) nitrogen.oxide< 0.657 77  1905.1950 17.81818 *
    ##      5) lstat< 14.4 189  4075.3230 22.94286  
    ##       10) lstat>=5.51 168  2309.9490 22.09702  
    ##         20) lstat>=9.705 88   616.5777 20.71591 *
    ##         21) lstat< 9.705 80  1340.8690 23.61625 *
    ##       11) lstat< 5.51 21   683.6381 29.70952 *
    ##    3) room>=6.8375 73  5959.9890 35.70411  
    ##      6) room< 7.443 49  2037.2070 31.28367  
    ##       12) lstat>=9.76 8   440.5750 23.02500 *
    ##       13) lstat< 9.76 41   944.5190 32.89512 *
    ##      7) room>=7.443 24  1010.4700 44.72917  
    ##       14) pt.ratio>=17.6 7   465.9686 38.88571 *
    ##       15) pt.ratio< 17.6 17   207.0588 47.13529 *

``` r
# predictions

prediction_DT <- model_DT %>% predict(test.data)

# Model performance

caret::R2(prediction_DT, test.data$house.value)
```

    ## [1] 0.7958708

``` r
caret::RMSE(prediction_DT, test.data$house.value)
```

    ## [1] 4.211489

Decision tree is a powerful machine learning algorithm. However, it do
come with its disadvantages. It is because that only one tree is built
and therefore its result is highly relied on the training set that used
in the split of the decision tree. Therefore, there might be significant
impact on the tree if there is a small change in the dataset. The lower
result above (79.58%) might be because that the decision tree was not
generalise well and overfit the training data.

To solve this problem, next section I will build and aggregate many
trees for do a better prediction, it is known as random forest. I will
hope to see a boost in prediction accuracy on the test dataset.

### 5.8 Random Forest

Random forest, is also known as an ensemble learning or method because
it aggregates many of decision trees, averaging the internally built
models, and creating a final high-performance predictive model compared
to decision tree.

This section will grow a high “data-diversity” random forest like many
other projects in the world that used Random forest. All trees will not
be completely the same. It is because of 2 popular techniques (James et
al. 2014, P. Bruce and Bruce 2017):

-   1.  each of the tree randomly grabs a data subset from the input
        training dataset and averaging the models’ results, it is known
        as *bagging* or *bootstrap aggregating*) and,

-   2.  each of the tree will be given randomly a smaller number of
        variables for them to choose when they do their split in their
        nodes, it is known as *bootstrap sampling*.

``` r
set.seed(123)

model_rf <- train(house.value ~., data = train.data,
               method = "rf",
               trControl = trainControl(method = "repeatedcv",
                                        number = 10,
                                        repeats = 3), 
               importance = T)

model_rf$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x)), importance = ..1) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 7
    ## 
    ##           Mean of squared residuals: 13.75115
    ##                     % Var explained: 83.6

By default, 500 trees were grown to build the random forest model with
their results averaged. The internal accuracy is 83.29%, which is
considered very good. You will see a much better at 89.9% (or 90%) later
when I predict on the test dataset.

The optimum number (mtry) of predictor variables randomly selected as
variables of choice during each split is 7. Ths value is automatically
discovered by the *caret* package.

``` r
model_rf$bestTune
```

    ##   mtry
    ## 2    7

The prediction result of this random forest model with the test data is:

``` r
# Make predictions

predictions <- model_rf %>% predict(test.data)

# Model performance

caret::R2(predictions, test.data$house.value)
```

    ## [1] 0.9030097

``` r
caret::RMSE(predictions, test.data$house.value)
```

    ## [1] 2.989836

Plotting the variance importance plot of random forest and having
following result. This “importance” plot tells you which variables are
importance in term of their predictive power in relative to each other.
The higher the importance ranking of one variable in the importance
plot, the higher its impact on the outcome variables. It is also
associated with an increase in significance level (lower P-value).

``` r
plot(varImp(model_rf))
```

![](boston_files/figure-gfm/unnamed-chunk-56-1.png)<!-- -->

This importance plot created by the random forest algorithm is useful in
telling us how far the important features are away from the unimportant
ones. I can see that “lstat” which is the proportion of lower status of
the population in the community and the number of room by “room” are the
two most important features in predicting the median house prices.

This result is the same as the results from multiple regression
regression in previous section 5.2, check out following summary tabels.

``` r
options(scipen = 999)

summary(model_mlr)$coef %>% 
  data.frame() %>% 
  rename(P_Value = Pr...t..) %>% 
  arrange(P_Value) %>% 
  dplyr::select(Estimate, P_Value) %>%  
  mutate(Significance.ordered = case_when(P_Value < 0.05 & P_Value > 0.01 ~ "* (< 0.05)",
                                  P_Value < 0.01 & P_Value > 0.001 ~ "** (< 0.01)",
                                  P_Value < 0.001 ~ "*** (< 0.001)",
                                  TRUE ~ " ")) %>% 
  arrange(P_Value) %>% 
  rownames_to_column(var = "features") %>%
  mutate("no." = row_number()) %>% 
  relocate("no.", .before = features) %>% 
  filter(features != "(Intercept)",
         Estimate > 0) %>% 
  kbl(caption = "Factors that Negatively Correlated with House Prices") %>% 
  kable_paper() 
```

<table class=" lightable-paper" style='font-family: "Arial Narrow", arial, helvetica, sans-serif; margin-left: auto; margin-right: auto;'>
<caption>
Factors that Negatively Correlated with House Prices
</caption>
<thead>
<tr>
<th style="text-align:right;">
no.
</th>
<th style="text-align:left;">
features
</th>
<th style="text-align:right;">
Estimate
</th>
<th style="text-align:right;">
P\_Value
</th>
<th style="text-align:left;">
Significance.ordered
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
room
</td>
<td style="text-align:right;">
3.6429412
</td>
<td style="text-align:right;">
0.0000000
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:left;">
highway.index
</td>
<td style="text-align:right;">
0.3256347
</td>
<td style="text-align:right;">
0.0000430
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:left;">
black
</td>
<td style="text-align:right;">
0.0109290
</td>
<td style="text-align:right;">
0.0002863
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:left;">
charles.river1
</td>
<td style="text-align:right;">
2.3375772
</td>
<td style="text-align:right;">
0.0153213
</td>
<td style="text-align:left;">

-   (&lt; 0.05)
    </td>
    </tr>
    <tr>
    <td style="text-align:right;">
    12
    </td>
    <td style="text-align:left;">
    resid.zone
    </td>
    <td style="text-align:right;">
    0.0385582
    </td>
    <td style="text-align:right;">
    0.0185896
    </td>
    <td style="text-align:left;">

    -   (&lt; 0.05)
        </td>
        </tr>
        <tr>
        <td style="text-align:right;">
        13
        </td>
        <td style="text-align:left;">
        age
        </td>
        <td style="text-align:right;">
        0.0048901
        </td>
        <td style="text-align:right;">
        0.7578730
        </td>
        <td style="text-align:left;">
        </td>
        </tr>
        </tbody>
        </table>

``` r
options(scipen = 1)
```

``` r
options(scipen = 999)

summary(model_mlr)$coef %>% 
  data.frame() %>% 
  rename(P_Value = Pr...t..) %>% 
  arrange(P_Value) %>% 
  dplyr::select(Estimate, P_Value) %>%  
  mutate(Significance.ordered = case_when(P_Value < 0.05 & P_Value > 0.01 ~ "* (< 0.05)",
                                  P_Value < 0.01 & P_Value > 0.001 ~ "** (< 0.01)",
                                  P_Value < 0.001 ~ "*** (< 0.001)",
                                  TRUE ~ " ")) %>% 
  arrange(P_Value) %>% 
  rownames_to_column(var = "features") %>%
  mutate("no." = row_number()) %>% 
  relocate("no.", .before = features) %>% 
  filter(features != "(Intercept)",
         Estimate < 0) %>% 
  kbl(caption = "Factors that Negatively Correlated with House Prices") %>% 
  kable_paper() 
```

<table class=" lightable-paper" style='font-family: "Arial Narrow", arial, helvetica, sans-serif; margin-left: auto; margin-right: auto;'>
<caption>
Factors that Negatively Correlated with House Prices
</caption>
<thead>
<tr>
<th style="text-align:right;">
no.
</th>
<th style="text-align:left;">
features
</th>
<th style="text-align:right;">
Estimate
</th>
<th style="text-align:right;">
P\_Value
</th>
<th style="text-align:left;">
Significance.ordered
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
lstat
</td>
<td style="text-align:right;">
-0.5167881
</td>
<td style="text-align:right;">
0.0000000
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:left;">
pt.ratio
</td>
<td style="text-align:right;">
-0.9692352
</td>
<td style="text-align:right;">
0.0000000
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:left;">
dist.to.work
</td>
<td style="text-align:right;">
-1.4098072
</td>
<td style="text-align:right;">
0.0000000
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:left;">
nitrogen.oxide
</td>
<td style="text-align:right;">
-17.3257148
</td>
<td style="text-align:right;">
0.0001090
</td>
<td style="text-align:left;">
\*\*\* (&lt; 0.001)
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:left;">
property.tax
</td>
<td style="text-align:right;">
-0.0123086
</td>
<td style="text-align:right;">
0.0057384
</td>
<td style="text-align:left;">
\*\* (&lt; 0.01)
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:left;">
crime.rate
</td>
<td style="text-align:right;">
-0.0977051
</td>
<td style="text-align:right;">
0.0151163
</td>
<td style="text-align:left;">

-   (&lt; 0.05)
    </td>
    </tr>
    <tr>
    <td style="text-align:right;">
    14
    </td>
    <td style="text-align:left;">
    indus.biz
    </td>
    <td style="text-align:right;">
    -0.0185927
    </td>
    <td style="text-align:right;">
    0.7935921
    </td>
    <td style="text-align:left;">
    </td>
    </tr>
    </tbody>
    </table>

``` r
options(scipen = 1)
```

-   Importance plot shows that “room” is the most important variable,
    which is supported by multiple linear regression summary with
    indication that this relationship is negative.

-   Importance plot shows that “lstat” is the second most important
    variable, which is supported by multiple linear regression summary
    with indication that this relationship is positive.

The results of these two different algorithms usually are not the same.
For example, the relationships between features and the responding
variables in this project is non-linear and the result from multiple
linear regression may not be so correct as compared to random forest
that can digest non-linearity. Results from both algorithm could be
similar (not entire the same) only if predictive power among features
are really outcompete others in a significant way.

Variables “lstat” and “room” are instead supported by the results of the
two different algorithms, and therefore it is robust to say “lstat” and
“room” are the two most important features in the prediction of median
house prices.

### 5.9 Stochastic gradient boosting (*XGBoost* in R)

This section builds an alternative forest, known as “XGBoost”. Compared
to random forest, many trees will also be grown in XGBoost “forest”.
However, trees are now grown sequentially, one by one, by using
information from previously grown trees with an aim to minimise the
error from the previous models (James et al. 2014). Therefore, trees
will become better and better with lesser and lesser error.

In the following, I am applying the *caret* package to combine with the
*xgboost* package to automatically find the best tuning parameters and
fit the final best boosted tree.

``` r
set.seed(123)

model_xgb <- train(house.value ~., data = train.data,
                   method = "xgbTree",
                   trControl = trainControl(method = "repeatedcv",
                                        number = 10,
                                        repeats = 3) 
                   )
```

This *caret* codes help to search a numbers of tuning parameters as
shown below.

``` r
model_xgb
```

    ## eXtreme Gradient Boosting 
    ## 
    ## 407 samples
    ##  13 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 366, 367, 366, 366, 366, 366, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   eta  max_depth  colsample_bytree  subsample  nrounds  RMSE      Rsquared 
    ##   0.3  1          0.6               0.50        50      4.189776  0.7851546
    ##   0.3  1          0.6               0.50       100      4.162309  0.7864778
    ##   0.3  1          0.6               0.50       150      4.229587  0.7780758
    ##   0.3  1          0.6               0.75        50      4.111238  0.7931788
    ##   0.3  1          0.6               0.75       100      4.092681  0.7941121
    ##   0.3  1          0.6               0.75       150      4.127679  0.7902403
    ##   0.3  1          0.6               1.00        50      4.156256  0.7879687
    ##   0.3  1          0.6               1.00       100      4.142841  0.7896297
    ##   0.3  1          0.6               1.00       150      4.143610  0.7896701
    ##   0.3  1          0.8               0.50        50      4.165157  0.7894113
    ##   0.3  1          0.8               0.50       100      4.087592  0.7948541
    ##   0.3  1          0.8               0.50       150      4.092606  0.7930074
    ##   0.3  1          0.8               0.75        50      4.123355  0.7906785
    ##   0.3  1          0.8               0.75       100      4.179660  0.7847117
    ##   0.3  1          0.8               0.75       150      4.199453  0.7832948
    ##   0.3  1          0.8               1.00        50      4.139836  0.7876808
    ##   0.3  1          0.8               1.00       100      4.131703  0.7885040
    ##   0.3  1          0.8               1.00       150      4.126364  0.7897973
    ##   0.3  2          0.6               0.50        50      3.803650  0.8195566
    ##   0.3  2          0.6               0.50       100      3.717671  0.8268153
    ##   0.3  2          0.6               0.50       150      3.697106  0.8283528
    ##   0.3  2          0.6               0.75        50      3.740200  0.8213224
    ##   0.3  2          0.6               0.75       100      3.591036  0.8339255
    ##   0.3  2          0.6               0.75       150      3.567242  0.8358048
    ##   0.3  2          0.6               1.00        50      3.755523  0.8224428
    ##   0.3  2          0.6               1.00       100      3.658675  0.8295833
    ##   0.3  2          0.6               1.00       150      3.606413  0.8338017
    ##   0.3  2          0.8               0.50        50      3.722845  0.8257211
    ##   0.3  2          0.8               0.50       100      3.574956  0.8395282
    ##   0.3  2          0.8               0.50       150      3.583798  0.8384354
    ##   0.3  2          0.8               0.75        50      3.645197  0.8364350
    ##   0.3  2          0.8               0.75       100      3.510652  0.8464859
    ##   0.3  2          0.8               0.75       150      3.496145  0.8473652
    ##   0.3  2          0.8               1.00        50      3.726654  0.8233437
    ##   0.3  2          0.8               1.00       100      3.639853  0.8306968
    ##   0.3  2          0.8               1.00       150      3.610958  0.8322943
    ##   0.3  3          0.6               0.50        50      3.669020  0.8321834
    ##   0.3  3          0.6               0.50       100      3.591623  0.8395536
    ##   0.3  3          0.6               0.50       150      3.575955  0.8405104
    ##   0.3  3          0.6               0.75        50      3.552543  0.8410422
    ##   0.3  3          0.6               0.75       100      3.526917  0.8424199
    ##   0.3  3          0.6               0.75       150      3.514443  0.8428980
    ##   0.3  3          0.6               1.00        50      3.574724  0.8391502
    ##   0.3  3          0.6               1.00       100      3.553550  0.8406611
    ##   0.3  3          0.6               1.00       150      3.556043  0.8403640
    ##   0.3  3          0.8               0.50        50      3.828091  0.8192159
    ##   0.3  3          0.8               0.50       100      3.790146  0.8225129
    ##   0.3  3          0.8               0.50       150      3.785898  0.8226046
    ##   0.3  3          0.8               0.75        50      3.548494  0.8390757
    ##   0.3  3          0.8               0.75       100      3.531651  0.8398979
    ##   0.3  3          0.8               0.75       150      3.532519  0.8397190
    ##   0.3  3          0.8               1.00        50      3.486991  0.8467347
    ##   0.3  3          0.8               1.00       100      3.480540  0.8466606
    ##   0.3  3          0.8               1.00       150      3.485283  0.8463671
    ##   0.4  1          0.6               0.50        50      4.198218  0.7860719
    ##   0.4  1          0.6               0.50       100      4.204524  0.7845740
    ##   0.4  1          0.6               0.50       150      4.171578  0.7877872
    ##   0.4  1          0.6               0.75        50      4.231983  0.7842167
    ##   0.4  1          0.6               0.75       100      4.216740  0.7814804
    ##   0.4  1          0.6               0.75       150      4.232249  0.7786622
    ##   0.4  1          0.6               1.00        50      4.172368  0.7841458
    ##   0.4  1          0.6               1.00       100      4.109850  0.7903851
    ##   0.4  1          0.6               1.00       150      4.115078  0.7897200
    ##   0.4  1          0.8               0.50        50      4.213517  0.7825986
    ##   0.4  1          0.8               0.50       100      4.170898  0.7873579
    ##   0.4  1          0.8               0.50       150      4.172120  0.7887917
    ##   0.4  1          0.8               0.75        50      4.121930  0.7922201
    ##   0.4  1          0.8               0.75       100      4.101577  0.7922773
    ##   0.4  1          0.8               0.75       150      4.101497  0.7915438
    ##   0.4  1          0.8               1.00        50      4.138594  0.7869596
    ##   0.4  1          0.8               1.00       100      4.131643  0.7887864
    ##   0.4  1          0.8               1.00       150      4.140788  0.7883835
    ##   0.4  2          0.6               0.50        50      3.885451  0.8112289
    ##   0.4  2          0.6               0.50       100      3.786123  0.8192151
    ##   0.4  2          0.6               0.50       150      3.739250  0.8229238
    ##   0.4  2          0.6               0.75        50      3.717788  0.8292076
    ##   0.4  2          0.6               0.75       100      3.616377  0.8363599
    ##   0.4  2          0.6               0.75       150      3.606505  0.8363261
    ##   0.4  2          0.6               1.00        50      3.802233  0.8151031
    ##   0.4  2          0.6               1.00       100      3.686314  0.8243742
    ##   0.4  2          0.6               1.00       150      3.663689  0.8259475
    ##   0.4  2          0.8               0.50        50      3.805126  0.8206570
    ##   0.4  2          0.8               0.50       100      3.736692  0.8250245
    ##   0.4  2          0.8               0.50       150      3.708495  0.8285183
    ##   0.4  2          0.8               0.75        50      3.691337  0.8299594
    ##   0.4  2          0.8               0.75       100      3.587630  0.8388444
    ##   0.4  2          0.8               0.75       150      3.558365  0.8411193
    ##   0.4  2          0.8               1.00        50      3.765799  0.8224197
    ##   0.4  2          0.8               1.00       100      3.618046  0.8336428
    ##   0.4  2          0.8               1.00       150      3.611047  0.8336760
    ##   0.4  3          0.6               0.50        50      3.812909  0.8267497
    ##   0.4  3          0.6               0.50       100      3.763888  0.8293300
    ##   0.4  3          0.6               0.50       150      3.775272  0.8287954
    ##   0.4  3          0.6               0.75        50      3.645958  0.8341149
    ##   0.4  3          0.6               0.75       100      3.622206  0.8368058
    ##   0.4  3          0.6               0.75       150      3.638841  0.8352735
    ##   0.4  3          0.6               1.00        50      3.637914  0.8321681
    ##   0.4  3          0.6               1.00       100      3.609429  0.8341381
    ##   0.4  3          0.6               1.00       150      3.613616  0.8337972
    ##   0.4  3          0.8               0.50        50      3.770588  0.8231466
    ##   0.4  3          0.8               0.50       100      3.770893  0.8223023
    ##   0.4  3          0.8               0.50       150      3.805073  0.8207812
    ##   0.4  3          0.8               0.75        50      3.612268  0.8347216
    ##   0.4  3          0.8               0.75       100      3.618751  0.8335765
    ##   0.4  3          0.8               0.75       150      3.603129  0.8343715
    ##   0.4  3          0.8               1.00        50      3.519945  0.8426974
    ##   0.4  3          0.8               1.00       100      3.531738  0.8418573
    ##   0.4  3          0.8               1.00       150      3.541093  0.8410112
    ##   MAE     
    ##   2.901790
    ##   2.838405
    ##   2.848366
    ##   2.812997
    ##   2.756037
    ##   2.736720
    ##   2.860127
    ##   2.797161
    ##   2.760498
    ##   2.899602
    ##   2.803154
    ##   2.782348
    ##   2.845007
    ##   2.823206
    ##   2.806186
    ##   2.847221
    ##   2.787556
    ##   2.762561
    ##   2.622077
    ##   2.578403
    ##   2.533551
    ##   2.550101
    ##   2.434470
    ##   2.405851
    ##   2.577328
    ##   2.464856
    ##   2.428544
    ##   2.550656
    ##   2.462248
    ##   2.477318
    ##   2.502441
    ##   2.393240
    ##   2.388367
    ##   2.524414
    ##   2.449682
    ##   2.424107
    ##   2.561106
    ##   2.510775
    ##   2.499400
    ##   2.427628
    ##   2.400694
    ##   2.396384
    ##   2.387770
    ##   2.373291
    ##   2.385525
    ##   2.630002
    ##   2.594515
    ##   2.589268
    ##   2.431589
    ##   2.417795
    ##   2.422249
    ##   2.383951
    ##   2.372599
    ##   2.373566
    ##   2.967161
    ##   2.916485
    ##   2.878347
    ##   2.923094
    ##   2.829805
    ##   2.798984
    ##   2.865730
    ##   2.772787
    ##   2.747468
    ##   2.978658
    ##   2.897110
    ##   2.856368
    ##   2.892791
    ##   2.807242
    ##   2.768749
    ##   2.873764
    ##   2.820475
    ##   2.801911
    ##   2.673810
    ##   2.605481
    ##   2.574590
    ##   2.577104
    ##   2.512968
    ##   2.506743
    ##   2.592469
    ##   2.516785
    ##   2.493486
    ##   2.622820
    ##   2.574954
    ##   2.556060
    ##   2.528548
    ##   2.442653
    ##   2.432378
    ##   2.521558
    ##   2.408921
    ##   2.400617
    ##   2.667463
    ##   2.638041
    ##   2.648155
    ##   2.502516
    ##   2.480522
    ##   2.486876
    ##   2.488359
    ##   2.471257
    ##   2.482270
    ##   2.596783
    ##   2.609937
    ##   2.639025
    ##   2.480500
    ##   2.493350
    ##   2.483111
    ##   2.400095
    ##   2.402896
    ##   2.414607
    ## 
    ## Tuning parameter 'gamma' was held constant at a value of 0
    ## Tuning
    ##  parameter 'min_child_weight' was held constant at a value of 1
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were nrounds = 100, max_depth = 3, eta
    ##  = 0.3, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1 and subsample
    ##  = 1.

The final values used for the model were nrounds = 100, max\_depth = 3,
eta = 0.4, gamma = 0, colsample\_bytree = 0.8, min\_child\_weight = 1
and subsample = 1.

``` r
model_xgb$bestTune
```

    ##    nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
    ## 53     100         3 0.3     0              0.8                1         1

Making prediction on the test data using this xgb\_model.

``` r
# predictions

predictions <- model_xgb %>% predict(test.data)

# model performance

caret::R2(predictions, test.data$house.value)
```

    ## [1] 0.8971142

``` r
caret::RMSE(predictions, test.data$house.value)
```

    ## [1] 3.005415

This model has results that are not too far away from random forest.
Next section will create a graph to compare the performance metrics of
all machine learning models I have trained.

### 6.0 Final Model Comparison

``` r
# set up dataframe

Model <- c("Lasso", 
           "PLS",
            "KNN",
           "Decision Tree",
           "Random Forest",
           "XGBoost")

R2_value <- c(0.7652148,
              0.7687147,
              0.8841925,
              0.7953629,
              0.8996801,
              0.8844266)
              
RMSE_value <- c(4.663365,
                4.531626,
                3.46252,
                4.217293,
                3.038207, 
                3.190433
                )

models <- data.frame(Model, R2_value, RMSE_value)

models <- models %>% 
  mutate(R2_percentage = round(R2_value*100, 2),
         Error_rate_percent = RMSE_value/mean(test.data$house.value),
         Error_rate_percent = round(Error_rate_percent * 100,2)) %>% 
  pivot_longer(c(4:5), names_to = "metrics", values_to = "results") %>% 
  mutate(Model = reorder_within(x = Model, by = results, within = metrics))
  

# plot

ggplot(models, aes(x = fct_reorder(Model, -results), y = results, fill = metrics)) +
  geom_bar(stat = "identity") +
  facet_wrap(~metrics, scale = "free_x") +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 10),
        plot.title = element_text(face = "bold")) +
  scale_x_reordered() +
  labs(x = "Models",
       y = "%",
       title = "Comparing Performance Metrics of All ML Models") +
  geom_text(aes(label = paste0(results, "%"), vjust = 1.5))
```

![](boston_files/figure-gfm/unnamed-chunk-63-1.png)<!-- -->

From this result, I conclude that random forest algorithm has the
highest prediction performance on the randomly sampled test dataset in
this project. It has the highest good R-squared value (R2, %) at 89.97%,
or 90%.

The 89.97% R-squared value means that the predicted outcome values by
the Random Forest using the test dataset has a high correlation with the
observed values in the test dataset. Alternatively, results are
approximately 90% similar. Random forest has also the lowest prediction
error rate at 13.43% (RMSE divided by the mean of the y variables in
test dataset).

## 6 Model for Production

This section uses RShiny to produce an online interactive application to
make predictions using the random forest algorithm.

Demo picture:

![](https://raw.githubusercontent.com/KAR-NG/Predicting-House-Prices-in-Boston_UniqueVersion/main/pic4_shiny.JPG)

**Visit this link to use the app:**

<https://karhou.shinyapps.io/boston/>

**Visit this github link to view the codes I used to program this app.**

<https://github.com/KAR-NG/Predicting-House-Prices-in-Boston_UniqueVersion/blob/main/app.R>

## 7 Conclusion

In conclusion, 7 different models were built to study this dataset,
included multiple linear regression (MLR), Lasso regression, Partial
Least Squares (PLS), K-Nearest Neighbor (KNN), Decision tree, random
forest, and stochastic gradient boosted random forest (XGBoost).

-   Random forest model had the best predictive power at 90% compared to
    all other models and should be used for prediction. A RShiny app has
    been built to make this model into production.

-   The variable “*age*” that stands for “Proportion of owner-occupied
    units built prior to 1940” and the variable “*indus*” that stands
    for “Proportion of non-retail business acres per town” are **not
    related** in the house prices with P higher than 0.05.

-   The **3 most positively related** variables are the *number of
    rooms* that affects house prices the most with the most significance
    level (lowest P-value with a value of &lt; 0.001), followed by the
    *proportion of black community* (P-value &lt;0.001) and the *index
    of accessibility to radial highway* (P-value &lt;0.001).

-   The **3 most negatively related** variables are the *lower status of
    the population* (percent) (P-value &lt;0.001), followed by the
    second ranked *nitrogen oxide concentration* (P-value &lt;0.001),
    and the third negative variable is *crime rate* (P-value &lt; 0.05).
    The higher the values of these variables, house prices are
    negatively impacted the most.

-   In overall, the most important 2 variables related to the Boston
    median house prices in the late 70s are the number of room (positive
    related) and the proportion of lower status population in the
    community (negative related).

*Thank you for reading*

## 8 LEGALITY

The purpose of this project is for educational and skills demonstration
ONLY.

## 9 REFERENCE

Boston thumbnail picture By King of Hearts - Own work, CC BY-SA 4.0,
<https://commons.wikimedia.org/w/index.php?curid=62981160>

Belsley D.A., Kuh, E. and Welsch, R.E. (1980) Regression Diagnostics.
Identifying Influential Data and Sources of Collinearity. New York:
Wiley.

Brownlee J 2016, *How to Work Through a Regression Machine Learning
Project in Weka*, viewed 26 September 2021,
<https://machinelearningmastery.com/regression-machine-learning-tutorial-weka/>

Data Professor 2020, *Web Apps in R: Building the Machine Learning Web
Application in R \| Shiny Tutorial Ep 4*, viewed 4 October 2021,
<https://www.youtube.com/watch?v=ceg7MMQNln8&t=847s>

Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand
for clean air. J. Environ. Economics and Management 5, 81–102.

James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani.
2014. An Introduction to Statistical Learning: With Applications in R .
Springer Publishing Company, Incorporated.

Minitab Blog Editor 2013, *Enough Is Enough! Handling Multicollinearity
in Regression Analysis*, viewed 25 September 2021,
<https://blog.minitab.com/en/understanding-statistics/handling-multicollinearity-in-regression-analysis>

Sivakumar C 2017, *<https://rpubs.com/chocka314/251613>*, viewed 28
September 2021, <https://rpubs.com/chocka314/251613>

<https://cran.r-project.org/web/packages/MASS/MASS.pdf>
