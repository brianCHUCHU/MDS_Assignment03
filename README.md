# Manufacturing Data Science Assignment 3
## Question 1:  Decision Tree Algorithms
Data Source: https://www.kaggle.com/uciml/faulty-steel-plates

Dataset provided by Semeion, Research Center of Sciences of Communication, Via Sersale 117, 00128, Rome, Italy. www.semeion.it

This dataset comes from research by Semeion, Research Center of Sciences of Communication. The original aim of the research was to correctly classify the type of surface defects in stainless steel plates, with six types of possible defects (plus "other"). The Input vector was made up of 27 indicators that approximately describe the geometric shape of the defect and its outline. 

There are 1941 plates with 34 variables. The first 27 columns (i.e. independent variables) describe some kind of steel plate faults seen in images, i.e., X1-X27, as 

{X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter,
Y_Perimeter, SumofLuminosity, MinimumofLuminosity, MaximumofLuminosity, LengthofConveyer, TypeOfSteel_A300, TypeOfSteel_A400, SteelPlateThickness, Edges_Index,Empty_Index, Square_Index, OutsideXIndex, EdgesXIndex, EdgesYIndex, OutsideGlobalIndex, LogOfAreas, LogXIndex, LogYIndex, Orientation_Index, Luminosity_Index, SigmoidOfAreas}


The last seven columns (i.e. dependent variables) are one hot encoded classes, i.e. if the plate fault is classified as "Stains" there will be a 1 in that column and 0's in the other columns.

{Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults}

These data can be found in http://archive.ics.uci.edu/ml/datasets/steel+plates+faults, and are
attached in the file MDS_Assignment3_Steelplates.xlsx.

### (a) Construct a data science framework and show the data summary

Data Science Framework

* Problem definition: 
   * Correctly classify the type of surface defects in stainless steel plates, with six types of possible defects (plus "other")
* Data Analysis: 
  * Data Preparation
    * Data Cleaning
    * Check Data Quality
    * Remove Redundant Column
  * Data Balancing
* Modeling
  * Model Structuring
    * CART
    * Random Forest
    * GBDT
  * Model Assessing
    * Cross Validation
* Decision Support: 
  * Model Criticism
  * Knowledge Management

Data Summary

1. Except for nominal variables like {TypeOfSteel_A300, TypeOfSteel_A400, , Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults}, all variables are numerical
2. The Summary of data is shown in the table below

|       |   X_Minimum |   X_Maximum |        Y_Minimum |        Y_Maximum |   Pixels_Areas |   X_Perimeter |   Y_Perimeter |   Sum_of_Luminosity |   Minimum_of_Luminosity |   Maximum_of_Luminosity |   Length_of_Conveyer |   TypeOfSteel_A300 |   TypeOfSteel_A400 |   Steel_Plate_Thickness |   Edges_Index |   Empty_Index |   Square_Index |   Outside_X_Index |   Edges_X_Index |   Edges_Y_Index |   Outside_Global_Index |   LogOfAreas |   Log_X_Index |   Log_Y_Index |   Orientation_Index |   Luminosity_Index |   SigmoidOfAreas |       Pastry |    Z_Scratch |    K_Scatch |       Stains |    Dirtiness |       Bumps |   Other_Faults |
|:------|------------:|------------:|-----------------:|-----------------:|---------------:|--------------:|--------------:|--------------------:|------------------------:|------------------------:|---------------------:|-------------------:|-------------------:|------------------------:|--------------:|--------------:|---------------:|------------------:|----------------:|----------------:|-----------------------:|-------------:|--------------:|--------------:|--------------------:|-------------------:|-----------------:|-------------:|-------------:|------------:|-------------:|-------------:|------------:|---------------:|
| count |    1941     |    1941     |   1941           |   1941           |        1941    |      1941     |      1941     |      1941           |               1941      |                1941     |             1941     |        1941        |        1941        |               1941      |   1941        |   1941        |    1941        |      1941         |     1941        |     1941        |            1941        |   1941       |   1941        |   1941        |        1941         |        1941        |      1941        | 1941         | 1941         | 1941        | 1941         | 1941         | 1941        |    1941        |
| mean  |     571.136 |     617.964 |      1.65068e+06 |      1.65074e+06 |        1893.88 |       111.855 |        82.966 |    206312           |                 84.5487 |                 130.194 |             1459.16  |           0.400309 |           0.599691 |                 78.7378 |      0.331715 |      0.414203 |       0.570767 |         0.0333611 |        0.610529 |        0.813472 |               0.575734 |      2.49239 |      1.33569  |      1.40327  |           0.0832876 |          -0.131305 |         0.58542  |    0.0814013 |    0.0978877 |    0.201443 |    0.0370943 |    0.0283359 |    0.20711  |       0.346728 |
| std   |     520.691 |     497.627 |      1.77458e+06 |      1.77459e+06 |        5168.46 |       301.209 |       426.483 |    512294           |                 32.1343 |                  18.691 |              144.578 |           0.490087 |           0.490087 |                 55.086  |      0.299712 |      0.137261 |       0.271058 |         0.0589612 |        0.243277 |        0.234274 |               0.482352 |      0.78893 |      0.481612 |      0.454345 |           0.500868  |           0.148767 |         0.339452 |    0.273521  |    0.297239  |    0.401181 |    0.189042  |    0.165973  |    0.405339 |       0.476051 |
| min   |       0     |       4     |   6712           |   6724           |           2    |         2     |         1     |       250           |                  0      |                  37     |             1227     |           0        |           0        |                 40      |      0        |      0        |       0.0083   |         0.0015    |        0.0144   |        0.0484   |               0        |      0.301   |      0.301    |      0        |          -0.991     |          -0.9989   |         0.119    |    0         |    0         |    0        |    0         |    0         |    0        |       0        |
| 25%   |      51     |     192     | 471253           | 471281           |          84    |        15     |        13     |      9522           |                 63      |                 124     |             1358     |           0        |           0        |                 40      |      0.0604   |      0.3158   |       0.3613   |         0.0066    |        0.4118   |        0.5968   |               0        |      1.9243  |      1        |      1.0792   |          -0.3333    |          -0.195    |         0.2482   |    0         |    0         |    0        |    0         |    0         |    0        |       0        |
| 50%   |     435     |     467     |      1.20413e+06 |      1.20414e+06 |         174    |        26     |        25     |     19202           |                 90      |                 127     |             1364     |           0        |           1        |                 70      |      0.2273   |      0.4121   |       0.5556   |         0.0101    |        0.6364   |        0.9474   |               1        |      2.2406  |      1.1761   |      1.3222   |           0.0952    |          -0.133    |         0.5063   |    0         |    0         |    0        |    0         |    0         |    0        |       0        |
| 75%   |    1053     |    1072     |      2.18307e+06 |      2.18308e+06 |         822    |        84     |        83     |     83011           |                106      |                 140     |             1650     |           1        |           1        |                 80      |      0.5738   |      0.5016   |       0.8182   |         0.0235    |        0.8      |        1        |               1        |      2.9149  |      1.5185   |      1.7324   |           0.5116    |          -0.0666   |         0.9998   |    0         |    0         |    0        |    0         |    0         |    0        |       1        |
| max   |    1705     |    1713     |      1.29877e+07 |      1.29877e+07 |      152655    |     10449     |     18152     |         1.15914e+07 |                203      |                 253     |             1794     |           1        |           1        |                300      |      0.9952   |      0.9439   |       1        |         0.8759    |        1        |        1        |               1        |      5.1837  |      3.0741   |      4.2587   |           0.9917    |           0.6421   |         1        |    1         |    1         |    1        |    1         |    1         |    1        |       1        |


3. Overall Distribution of the data set is shown below, many of columns are not normally distributed, a model that would not be affected by distribution (e.g. Decision Tree) would be preferred.

![Alt text](image.png)


### (b) What is the problem about the dataset? Any identical column? Any redundant column? Any missing value? How to handle these issues?
* No missing value in this dataset
* By Observing the correlation:
  * Type of steel: should be combined to one column, since it is actually one-hot encoded.
  * X_min, X_max, Y_min, Y_max: Those are highly correlated between the two pairs.Therefore, it is combined to two columns: mean and relative percent range.
  * For Pixel_Areas, XY_Perimeters, LogofAreas, Log_XY_Index: Those variables are also highly correlated and may have collinearity, but sometimes redundant columns may help the model, and meaning of those variables remain unknown due to lack of domain knowledge. We'll come to it if the model is underperformed.
  * Concat Classes: Combine one-hot encoded y variable into one column in order to fit the model

![Alt text](image-1.png)
![Alt text](image-2.png)

### (c) After data preprocessing, based on the prepared dataset, use the classification and regression tree (CART) to analyze the prepared dataset. Show the classification results by 10-fold cross validation with several metrics (eg. accuracy, area under ROC curve (AUC), and F1-score), and also list the hyperparameters you adjust.

* Hyperparameters of CART:
  * max_depth=9
  * min_samples_split=27m

|    |   fit_time |   score_time |   test_accuracy |   test_roc_auc_ovr |   test_f1_micro |
|---:|-----------:|-------------:|----------------:|-------------------:|----------------:|
|  0 |  0.0502644 |   0.00807333 |        0.45641  |           0.778946 |        0.45641  |
|  1 |  0.0479693 |   0.0080297  |        0.716495 |           0.83284  |        0.716495 |
|  2 |  0.0550723 |   0.008008   |        0.541237 |           0.887343 |        0.541237 |
|  3 |  0.0512764 |   0.00999618 |        0.639175 |           0.841487 |        0.639175 |
|  4 |  0.0496716 |   0.00799894 |        0.505155 |           0.848203 |        0.505155 |
|  5 |  0.0540152 |   0.00800037 |        0.582474 |           0.888918 |        0.582474 |
|  6 |  0.0529988 |   0.00999904 |        0.685567 |           0.920694 |        0.685567 |
|  7 |  0.0525146 |   0.0090003  |        0.726804 |           0.920822 |        0.726804 |
|  8 |  0.0529995 |   0.00700092 |        0.494845 |           0.902196 |        0.494845 |
|  9 |  0.0527313 |   0.00900674 |        0.556701 |           0.687926 |        0.556701 |

### (d) Suggest a method to address the data imbalance issue. Build a new balanced dataset. (hint: undersampling or oversampling)

Use imlearn package to import Near Miss Undersampling, while the method I use is "Majority class examples with minimum average distance to three closest minority class examples".


### (e) Based on the balanced dataset, use the classification and regression tree (CART) to analyze the balanced dataset. Show the classification results by 10-fold cross validation with several metrics (eg. accuracy, area under ROC curve (AUC), and F1-score), and also list the hyperparameters you adjust.

* Hyperparameters of CART:
  * max_depth=7
  * min_samples_split=25

|    |   fit_time |   score_time |   test_accuracy |   test_roc_auc_ovr |   test_f1_micro |
|---:|-----------:|-------------:|----------------:|-------------------:|----------------:|
|  0 | 0.0165098  |   0.00858521 |        0.717949 |           0.868054 |        0.717949 |
|  1 | 0.0162735  |   0.00700498 |        0.769231 |           0.87709  |        0.769231 |
|  2 | 0.00968313 |   0.00657845 |        0.564103 |           0.847683 |        0.564103 |
|  3 | 0.0160925  |   0.00791597 |        0.666667 |           0.870278 |        0.666667 |
|  4 | 0.00800014 |   0.00799966 |        0.74359  |           0.916166 |        0.74359  |
|  5 | 0.00800085 |   0.00799918 |        0.736842 |           0.91896  |        0.736842 |
|  6 | 0.016011   |   0.00798607 |        0.842105 |           0.959727 |        0.842105 |
|  7 | 0.00800323 |   0.00799823 |        0.842105 |           0.937534 |        0.842105 |
|  8 | 0.0160017  |   0.00800848 |        0.789474 |           0.943987 |        0.789474 |
|  9 | 0.00798845 |   0.0160005  |        0.815789 |           0.933015 |        0.815789 |

### (f) Give a comparison between (c) and (e). Any suggestion or insight?

The "prepared dataset" is significantly imbalanced, leading to overfitting on certain training folds. The "balanced dataset" after undersampling results in training a better overall model without overfitting issue.

### (g) Use “Random Forest” to solve both prepared dataset and balanced dataset, respectively. Give a comparison and provide your insight.

For prepared dataset, Random Forest model does not improve the performance much comparing to CART. But for balanced dataset, it improves a lot by fitting it into Random Forest. A balanced dataset are more likely to acheive higher feasibility in different random state. Besides from performance, its training efficiency also increased significantly (reduced roughly 70% of time), due to less samples after underfitting.

Prepared dataset:
* Hyperparameters of Random Forest:
  * max_depth=7
  * min_samples_split=25

|    |   fit_time |   score_time |   test_accuracy |   test_roc_auc_ovr |   test_f1_micro |
|---:|-----------:|-------------:|----------------:|-------------------:|----------------:|
|  0 |   0.930523 |    0.0291448 |        0.476923 |           0.804555 |        0.476923 |
|  1 |   0.967804 |    0.0240827 |        0.716495 |           0.945626 |        0.716495 |
|  2 |   0.994527 |    0.0252149 |        0.489691 |           0.888683 |        0.489691 |
|  3 |   0.936761 |    0.0243795 |        0.649485 |           0.904586 |        0.649485 |
|  4 |   0.956621 |    0.0177591 |        0.64433  |           0.907906 |        0.64433  |
|  5 |   0.957804 |    0.0160813 |        0.587629 |           0.88965  |        0.587629 |
|  6 |   0.957247 |    0.0256846 |        0.731959 |           0.926415 |        0.731959 |
|  7 |   0.922012 |    0.0238264 |        0.737113 |           0.952092 |        0.737113 |
|  8 |   1.00772  |    0.0172977 |        0.634021 |           0.938404 |        0.634021 |
|  9 |   0.95536  |    0.0160058 |        0.592784 |           0.837092 |        0.592784 |

Balanced dataset:
* No additional Hyperparameters of Random Forest

|    |   fit_time |   score_time |   test_accuracy |   test_roc_auc_ovr |   test_f1_micro |
|---:|-----------:|-------------:|----------------:|-------------------:|----------------:|
|  0 |   0.291687 |    0.0160072 |        0.666667 |           0.942569 |        0.666667 |
|  1 |   0.286062 |    0.0245056 |        0.794872 |           0.932183 |        0.794872 |
|  2 |   0.279644 |    0.023752  |        0.820513 |           0.976441 |        0.820513 |
|  3 |   0.277963 |    0.0207012 |        0.846154 |           0.963424 |        0.846154 |
|  4 |   0.322167 |    0.0239997 |        0.794872 |           0.969909 |        0.794872 |
|  5 |   0.304034 |    0.0164847 |        0.763158 |           0.934585 |        0.763158 |
|  6 |   0.304006 |    0.024606  |        0.894737 |           0.991518 |        0.894737 |
|  7 |   0.287461 |    0.0169592 |        0.894737 |           0.968764 |        0.894737 |
|  8 |   0.307669 |    0.0161641 |        0.842105 |           0.976339 |        0.842105 |
|  9 |   0.293931 |    0.0245423 |        0.710526 |           0.970326 |        0.710526 |

### (h) (5%) Use “Gradient Boosting Decision Tree (GBDT)” to solve both prepared dataset and balanced dataset, respectively. Give a comparison and provide your insight

The balanced dataset also improves the performance a lot in using the GBDT model as the Random Forest model did. But the balanced dataset improves the training efficiency more (reduced up to 90% time). The result implies that the sample num increases GBDT training time significantly.

Prepared dataset:

|    |   fit_time |   score_time |   test_accuracy |   test_roc_auc_ovr |   test_f1_micro |
|---:|-----------:|-------------:|----------------:|-------------------:|----------------:|
|  0 |    27.7761 |   0.0487523  |        0.497436 |           0.802356 |        0.497436 |
|  1 |    37.0345 |   0.0399592  |        0.670103 |           0.930282 |        0.670103 |
|  2 |    45.9873 |   0.0398364  |        0.520619 |           0.89089  |        0.520619 |
|  3 |    39.0748 |   0.0418594  |        0.670103 |           0.910219 |        0.670103 |
|  4 |    38.4939 |   0.0408568  |        0.695876 |           0.907127 |        0.695876 |
|  5 |    48.5071 |   0.0409379  |        0.587629 |           0.891235 |        0.587629 |
|  6 |    21.9736 |   0.0132005  |        0.773196 |           0.941301 |        0.773196 |
|  7 |    35.1309 |   0.01652    |        0.737113 |           0.959057 |        0.737113 |
|  8 |    16.5713 |   0.00808501 |        0.618557 |           0.937846 |        0.618557 |
|  9 |    16.5036 |   0.0159249  |        0.628866 |           0.821731 |        0.628866 |

Balanced dataset:

|    |   fit_time |   score_time |   test_accuracy |   test_roc_auc_ovr |   test_f1_micro |
|---:|-----------:|-------------:|----------------:|-------------------:|----------------:|
|  0 |    4.30198 |   0.00824642 |        0.641026 |           0.930193 |        0.641026 |
|  1 |    4.24922 |   0.00844693 |        0.769231 |           0.93498  |        0.769231 |
|  2 |    4.25328 |   0.0162177  |        0.794872 |           0.94868  |        0.794872 |
|  3 |    4.24151 |   0.0160885  |        0.794872 |           0.951082 |        0.794872 |
|  4 |    4.29585 |   0.00861716 |        0.871795 |           0.978847 |        0.871795 |
|  5 |    4.23032 |   0.00792933 |        0.763158 |           0.947267 |        0.763158 |
|  6 |    4.21367 |   0.016001   |        0.842105 |           0.987622 |        0.842105 |
|  7 |    4.23138 |   0.0160041  |        0.894737 |           0.9699   |        0.894737 |
|  8 |    4.25179 |   0.00800967 |        0.842105 |           0.963839 |        0.842105 |
|  9 |    4.2533  |   0.00973129 |        0.736842 |           0.97174  |        0.736842 |

## Question 2: Feature Selection and Regularization- Ridge, Lasso, and Elastic Net

Data Source: a flotation plant in a mining process

https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process 

Dataset provided by EduardoMagalhãesOliveira. Data collection methodology is shown as follows.

Hardware sensors, like temperature, pH, flow, density and all continuous process variables, where data were collected every 20s with no transformation (the dataset here shows raw data). Quality variables, like % of silica content, % of iron ore content and so on are quality measurements made by laboratory analysis. A sample of the iron ore pulp is collected in the
field/shop floor, every 15 minutes. Those samples are sent to lab for analysis. So, on every two
hours, lab give a feedback of quality analysis, in other words, only every two hours you have a
lab/quality measurement of the product stream (iron ore concentrate), which gives you a sense
of the quality of the product (iron ore pulp concentrate).

The main goal is to use this data to predict how much impurity is in the ore concentrate. As this
impurity is measured every hour, if we can predict how much silica (impurity) is in the ore
concentrate, we can help the engineers, giving them early information to take actions
(empowering!). Hence, they will be able to take corrective actions in advance (reduce impurity,
if it is the case) and also help the environment (reducing the amount of ore that goes to tailings
as you reduce silica in the ore concentrate).

### Content

The first column shows time and date range (from march of 2017 until september of 2017).
Some columns were sampled every 20 second. Others were sampled on a hourly base.
The second and third columns are quality measures of the iron ore pulp right before it is fed into
the flotation plant. Column 4 until column 8 are the most important variables that impact in the
ore quality in the end of the process. From column 9 until column 22, we can see process data
(level and air flow inside the flotation columns, which also impact in ore quality. The last two
columns are the final iron ore pulp quality measurement from the lab.)

Target is to predict the last column, which is the % of silica in the iron ore concentrate.

### Inspiration

Is it possible to predict % Silica Concentrate every minute?

How many steps (hours) ahead can we predict % Silica in Concentrate? This would help
engineers to act in predictive and optimized way, mitigating the % of iron that could have gone
to tailings.

Also, for the Amina Flow, Ore Pulp Flow, and Flotation Column, etc. data are the “commas” in
those cells supposed to be “decimals” as shown in the csv file. For example, given the column
“% Silica Concentrate”=”1,31”, it means that the concentrate is 1.31%.

According to the description mentioned above, if the factor “% Silica Concentrate” is
regarded as the response variable (y) and all factors (except the date and % Iron Concentrate)
are independent variables, how to identify the importance variable which significantly affects
the “% Silica Concentrate” (y)?

### (a) Identify the important variable by linear regression with ordinary least squares (OLS) (i.e. ranked by p-value). Identify the important variable by stepwise regression. (hint: you can select forward selection, backward elimination, or both)

Linear Regression (OLS): Ranked important variables are shown below, resulted in low R-squared(0.023) and influenced by collinearity.

| Variable                     |      p-value |
|:-----------------------------|-------------:|
| Flotation Column 03 Air Flow | 0            |
| Flotation Column 03 Level    | 0            |
| Flotation Column 06 Air Flow | 4.95701e-200 |
| Flotation Column 05 Air Flow | 1.05914e-139 |
| Ore Pulp pH                  | 1.96472e-101 |
| Flotation Column 04 Level    | 6.62629e-100 |
| Ore Pulp Flow                | 3.54601e-94  |
| Flotation Column 04 Air Flow | 2.56486e-86  |
| Starch Flow                  | 5.56344e-68  |
| Flotation Column 07 Air Flow | 9.80908e-51  |
| Flotation Column 01 Air Flow | 1.20653e-42  |
| Amina Flow                   | 1.15434e-31  |
| Flotation Column 01 Level    | 1.7032e-22   |
| Flotation Column 02 Level    | 1.74766e-20  |
| Flotation Column 07 Level    | 5.60645e-15  |
| Flotation Column 02 Air Flow | 1.28001e-10  |
| Flotation Column 05 Level    | 6.01838e-08  |
| Flotation Column 06 Level    | 0.00647916   |
| const                        | 0.00663494   |
| Ore Pulp Density             | 0.0645006    |
| % Silica Feed                | 0.356558     |
| % Iron Feed                  | 0.630777     |

Stepwise Regression: Use forward selection to choose 10 variables.

* top 10 variables: `('const', 'Starch Flow', 'Amina Flow', 'Ore Pulp pH', 'Ore Pulp Density', 'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow', 'Flotation Column 03 Level')`
* R-squared: -0.05935763531916578

```
                             OLS Regression Results                             
================================================================================
Dep. Variable:     % Silica Concentrate   R-squared:                       0.023
Model:                              OLS   Adj. R-squared:                  0.023
Method:                   Least Squares   F-statistic:                     879.8
Date:                  Thu, 16 Nov 2023   Prob (F-statistic):               0.00
Time:                          16:19:19   Log-Likelihood:            -1.8039e+07
No. Observations:                737453   AIC:                         3.608e+07
Df Residuals:                    737432   BIC:                         3.608e+07
Df Model:                            20                                         
Covariance Type:              nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                          1.82e+09    6.7e+08      2.715      0.007    5.06e+08    3.13e+09
% Iron Feed                   4.798e+06   9.98e+06      0.481      0.631   -1.48e+07    2.44e+07
% Silica Feed                 6.901e+06   7.48e+06      0.922      0.357   -7.77e+06    2.16e+07
Starch Flow                  -4.541e-05   2.61e-06    -17.424      0.000   -5.05e-05   -4.03e-05
Amina Flow                      -0.0002   2.06e-05    -11.709      0.000      -0.000      -0.000
Ore Pulp Flow                   -0.0002   9.35e-06    -20.591      0.000      -0.000      -0.000
Ore Pulp pH                      0.0206      0.001     21.385      0.000       0.019       0.022
Ore Pulp Density                -0.0067      0.004     -1.849      0.065      -0.014       0.000
Flotation Column 01 Air Flow    -0.0013   9.65e-05    -13.688      0.000      -0.002      -0.001
Flotation Column 02 Air Flow    -0.0005   7.68e-05     -6.430      0.000      -0.001      -0.000
Flotation Column 03 Air Flow     0.0046   8.65e-05     52.965      0.000       0.004       0.005
Flotation Column 04 Air Flow     0.0004   1.83e-05     19.694      0.000       0.000       0.000
Flotation Column 05 Air Flow     0.0005   1.86e-05     25.167      0.000       0.000       0.001
Flotation Column 06 Air Flow    -0.0022   7.37e-05    -30.185      0.000      -0.002      -0.002
Flotation Column 07 Air Flow    -0.0008    5.4e-05    -14.982      0.000      -0.001      -0.001
Flotation Column 01 Level     8.893e-05   9.11e-06      9.758      0.000    7.11e-05       0.000
Flotation Column 02 Level    -7.673e-05   8.27e-06     -9.277      0.000   -9.29e-05   -6.05e-05
Flotation Column 03 Level        0.0004   7.06e-06     53.935      0.000       0.000       0.000
Flotation Column 04 Level       -0.0004   1.92e-05    -21.220      0.000      -0.000      -0.000
Flotation Column 05 Level       -0.0001   1.87e-05     -5.418      0.000      -0.000   -6.46e-05
Flotation Column 06 Level    -4.803e-05   1.76e-05     -2.722      0.006   -8.26e-05   -1.35e-05
Flotation Column 07 Level        0.0002   2.08e-05      7.813      0.000       0.000       0.000
==============================================================================
Omnibus:                   526353.496   Durbin-Watson:                   0.175
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5629779.048
Skew:                           3.552   Prob(JB):                         0.00
Kurtosis:                      14.521   Cond. No.                     3.06e+14
==============================================================================
```

### (b) Give a comparison between OLS and Stepwise Regression. The results are consistant?

First we conduct a ols using only top 10 variables, resulting in 0.022 R-squared. Comparing to Stepwise Regression, it performs slightly better, but both of them are underperformed due to collinearity. The important variables are also not consistant, since one evaluate the model considering all columns and another improves it step by step.

```
                             OLS Regression Results                             
================================================================================
Dep. Variable:     % Silica Concentrate   R-squared:                       0.022
Model:                              OLS   Adj. R-squared:                  0.022
Method:                   Least Squares   F-statistic:                     1683.
Date:                  Thu, 16 Nov 2023   Prob (F-statistic):               0.00
Time:                          16:21:43   Log-Likelihood:            -1.8040e+07
No. Observations:                737453   AIC:                         3.608e+07
Df Residuals:                    737442   BIC:                         3.608e+07
Df Model:                            10                                         
Covariance Type:              nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                         2.171e+09   1.46e+07    148.312      0.000    2.14e+09     2.2e+09
Flotation Column 03 Air Flow     0.0032   4.83e-05     67.017      0.000       0.003       0.003
Flotation Column 03 Level        0.0004    6.3e-06     56.732      0.000       0.000       0.000
Flotation Column 06 Air Flow    -0.0026   6.88e-05    -37.466      0.000      -0.003      -0.002
Flotation Column 05 Air Flow     0.0005   1.86e-05     25.553      0.000       0.000       0.001
Ore Pulp pH                      0.0209      0.001     21.877      0.000       0.019       0.023
Flotation Column 04 Level       -0.0005   1.41e-05    -33.946      0.000      -0.001      -0.000
Ore Pulp Flow                   -0.0002   8.96e-06    -24.438      0.000      -0.000      -0.000
Flotation Column 04 Air Flow     0.0004   1.82e-05     19.672      0.000       0.000       0.000
Starch Flow                  -4.913e-05   2.56e-06    -19.179      0.000   -5.42e-05   -4.41e-05
Flotation Column 07 Air Flow    -0.0008   5.35e-05    -15.497      0.000      -0.001      -0.001
==============================================================================
Omnibus:                   526832.303   Durbin-Watson:                   0.173
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5645102.708
Skew:                           3.556   Prob(JB):                         0.00
Kurtosis:                      14.538   Cond. No.                     6.46e+12
==============================================================================
```

### (c) From a methodology aspect, what’s the difference between Ridge regression and Lasso? Why does Lasso support the variables selection rather than ridge? (hint: answer with description or formulation. No computation needed.)

Lasso tends to reduce one of the variable to zero first, while Ridge would only reduce every variable to zero when lambda is infinite. From the geometric point of view, Lasso represent a square area while Ridge would plot a round area.

### (d) What’s the benefit to use the Elastic Net? (hint: answer with description or formulation. No computation needed.)

It combined Lasso and Ridge, which results in that highly correlated covariates tend to have similar regression
coefficients.

### (e) Identify the important variable by ridge regression, lasso, and elastic net.

|Importance Ranking| Ridge Variables              | Lasso Variables              | Elastic Net Variables        |
|---:|:-----------------------------|:-----------------------------|:-----------------------------|
|  0 | % Iron Feed                  | % Iron Feed                  | % Iron Feed                  |
|  1 | % Silica Feed                | % Silica Feed                | % Silica Feed                |
|  2 | Ore Pulp Density             | Ore Pulp Density             | Ore Pulp Density             |
|  3 | % Iron Concentrate           | % Iron Concentrate           | % Iron Concentrate           |
|  4 | Ore Pulp pH                  | Ore Pulp pH                  | Ore Pulp pH                  |
|  5 | Flotation Column 03 Air Flow | Flotation Column 03 Air Flow | Flotation Column 03 Air Flow |
|  6 | Flotation Column 06 Air Flow | Flotation Column 06 Air Flow | Flotation Column 06 Air Flow |
|  7 | Flotation Column 02 Air Flow | Flotation Column 02 Air Flow | Flotation Column 02 Air Flow |
|  8 | Flotation Column 07 Air Flow | Flotation Column 07 Air Flow | Flotation Column 07 Air Flow |
|  9 | Flotation Column 07 Level    | Flotation Column 07 Level    | Flotation Column 07 Level    |
| 10 | Flotation Column 04 Level    | Flotation Column 04 Level    | Flotation Column 04 Level    |
| 11 | Flotation Column 06 Level    | Flotation Column 06 Level    | Flotation Column 06 Level    |
| 12 | Flotation Column 05 Air Flow | Flotation Column 05 Air Flow | Flotation Column 05 Air Flow |
| 13 | Flotation Column 01 Level    | Flotation Column 01 Level    | Flotation Column 01 Level    |
| 14 | Flotation Column 04 Air Flow | Flotation Column 04 Air Flow | Flotation Column 01 Air Flow |
| 15 | Flotation Column 01 Air Flow | Flotation Column 01 Air Flow | Flotation Column 04 Air Flow |
| 16 | Flotation Column 05 Level    | Flotation Column 05 Level    | Flotation Column 05 Level    |
| 17 | Flotation Column 02 Level    | Flotation Column 02 Level    | Flotation Column 02 Level    |
| 18 | Amina Flow                   | Amina Flow                   | Amina Flow                   |
| 19 | Starch Flow                  | Starch Flow                  | Starch Flow                  |
| 20 | Ore Pulp Flow                | Ore Pulp Flow                | Ore Pulp Flow                |
| 21 | Flotation Column 03 Level    | Flotation Column 03 Level    | Flotation Column 03 Level    |

(The Ranking is Ranked by RFE)

### (f) Give a comparison in (e). The results are consistent? If no, what’s the difference?

The result is consistent, expect for a few low-ranking variables of Elactic Net, but the order of the rest are identical.

### (g) What is “adaptive elastic net”? Why we need it? How to build it? Please simply describe or formulate it. (No computation needed.)

Used weighted L1 penalty, which can give larger penalty to the near-zero-coefficient variables (increase penalty over iteration). In this case, the model can result in a smaller bias.

### (h) Which columns are highly-correlated? Show the table of the coefficient estimation by using linear regression. Any multicollinearity problem?

```
                             OLS Regression Results                             
================================================================================
Dep. Variable:     % Silica Concentrate   R-squared:                       0.023
Model:                              OLS   Adj. R-squared:                  0.023
Method:                   Least Squares   F-statistic:                     879.8
Date:                  Thu, 16 Nov 2023   Prob (F-statistic):               0.00
Time:                          16:19:19   Log-Likelihood:            -1.8039e+07
No. Observations:                737453   AIC:                         3.608e+07
Df Residuals:                    737432   BIC:                         3.608e+07
Df Model:                            20                                         
Covariance Type:              nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                          1.82e+09    6.7e+08      2.715      0.007    5.06e+08    3.13e+09
% Iron Feed                   4.798e+06   9.98e+06      0.481      0.631   -1.48e+07    2.44e+07
% Silica Feed                 6.901e+06   7.48e+06      0.922      0.357   -7.77e+06    2.16e+07
Starch Flow                  -4.541e-05   2.61e-06    -17.424      0.000   -5.05e-05   -4.03e-05
Amina Flow                      -0.0002   2.06e-05    -11.709      0.000      -0.000      -0.000
Ore Pulp Flow                   -0.0002   9.35e-06    -20.591      0.000      -0.000      -0.000
Ore Pulp pH                      0.0206      0.001     21.385      0.000       0.019       0.022
Ore Pulp Density                -0.0067      0.004     -1.849      0.065      -0.014       0.000
Flotation Column 01 Air Flow    -0.0013   9.65e-05    -13.688      0.000      -0.002      -0.001
Flotation Column 02 Air Flow    -0.0005   7.68e-05     -6.430      0.000      -0.001      -0.000
Flotation Column 03 Air Flow     0.0046   8.65e-05     52.965      0.000       0.004       0.005
Flotation Column 04 Air Flow     0.0004   1.83e-05     19.694      0.000       0.000       0.000
Flotation Column 05 Air Flow     0.0005   1.86e-05     25.167      0.000       0.000       0.001
Flotation Column 06 Air Flow    -0.0022   7.37e-05    -30.185      0.000      -0.002      -0.002
Flotation Column 07 Air Flow    -0.0008    5.4e-05    -14.982      0.000      -0.001      -0.001
Flotation Column 01 Level     8.893e-05   9.11e-06      9.758      0.000    7.11e-05       0.000
Flotation Column 02 Level    -7.673e-05   8.27e-06     -9.277      0.000   -9.29e-05   -6.05e-05
Flotation Column 03 Level        0.0004   7.06e-06     53.935      0.000       0.000       0.000
Flotation Column 04 Level       -0.0004   1.92e-05    -21.220      0.000      -0.000      -0.000
Flotation Column 05 Level       -0.0001   1.87e-05     -5.418      0.000      -0.000   -6.46e-05
Flotation Column 06 Level    -4.803e-05   1.76e-05     -2.722      0.006   -8.26e-05   -1.35e-05
Flotation Column 07 Level        0.0002   2.08e-05      7.813      0.000       0.000       0.000
==============================================================================
Omnibus:                   526353.496   Durbin-Watson:                   0.175
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          5629779.048
Skew:                           3.552   Prob(JB):                         0.00
Kurtosis:                      14.521   Cond. No.                     3.06e+14
==============================================================================
```

By observing the table, the Durbin-Watson Statistic is only 0.175, implying autocorrelation. We therfore construct the Correlation Coeffient Table, resulting in `'% Iron Feed'` and `'% Silica Feed'` is highly negatively correlated.

![Alt text](image-3.png)

(i) (5%) Is it possible to predict % Silica in Concentrate with using % Iron Concentrate column
(as they are highly correlated)? Why? What’s the potential issue? How to address it?

The potential issue of high correlation is collineartiy, resulting in a tendacy to create singular matrices and other negative impact to the model.

But Yes, it is possible if we use embedded methods like Lasso or Elastic Net. It considered the interaction and non-linear relationship between variables.

## Question 3: Deep Learning

Use Python to build up long short-term memory (LSTM), which is one type of recurrent neural network (RNN). Collect the dataset related to weekly raw material price OR consumption (i.e. demand). Build a price/demand forecast. Don’t use STOCK PRICE for prediction. You may read the tutorial: https://www.datacamp.com/community/tutorials/lstm-python-stock-market. Note that, you only have price/demand data as response variable Y and it should be a time-rolling prediction, that is, for example, use the past 8 weeks dataset for 8-week ahead prediction. Thus, the prediction should be like the following diagram.

![Alt text](image-4.png)

Dataset could be found as follows.

eg. Brent oil price: https://www.investing.com/commodities/brent-oil-historical-data

Commodity prices: https://fred.stlouisfed.org/categories/32217

Commodity prices: https://sdw.ecb.europa.eu/browse.do?node=9691219

The summary table of raw materials, https://just2.entrust.com.tw/z/ze/zeq/zeq.djhtm

Pick one raw material and collect its dataset. The collection period should be as long as possible (eg. from 2000 to 2022) to guarantee the sufficient samples for LSTM training.

### (a) Prepare and transform the data to appropriate format (eg. use Data Generator in https://www.datacamp.com/community/tutorials/lstm-python-stock-market). Build LSTM model and show the prediction results via Time-series (Nested) Cross Validation.

Below is linechart of the Crude Oil Price Dataset.

![Alt text](image-5.png)


### (b) Visualize the time-rolling prediction as above diagram.
The LSTM prediction process:
![Alt text](image-6.png)