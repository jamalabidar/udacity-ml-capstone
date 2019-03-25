# Setup:

Python 3

The following python packages need to be installed:
- pandas
- numpy
- matplotlib
- seaborn
- xlrd
- scikit-learn
- statsmodels
- keras

All packages are in requirements.txt file

run: pip install -r requirements.txt


# Capstone Project: Predictive Maintenance

In this notebook, we'll try to solve a predictive maintenance problem on Kaggle: [Kaggle Predictive Maintennace Competition](https://www.kaggle.com/c/predictive-maintenance1).

The dataset provides us with some error logs for an equipment: error counts, times of occurence...

The dataset consists of 2 files:
- feature.xlsx
- train_label.csv

The excel file has 983 lines. One line for each day. The covered time period is from 5/3/2015 (3rd of April 2015) to 1/17/2018 (17th of January 2018).

In the feature.csv file, for each day we have 5 sets of 26 features:
- errors counts (number of errors that day) for 26 types of errors
- errors max ticks (time of the last error of the day) for 26 types of errors
- errors min ticks (time of the first error of the day) for 26 types of errors
- errors mean tick (mean time for the errors of the day) for 26 types of errors
- errors standard deviation of the ticks for 26 types of errors

Total number of features in the feature.xlsx file is 26 * 5 + 1(date) = 131.

The train_label.csv file has 683 lines. One line for each day which presents only one label: 0 when equipment is OK, 1 when equipment failed that day. the covered period is from 5/3/2015 (3rd of April 2015) to 3/24/2017 (24th of March 2017).

The objective it to use the provided data to predict when the equipment will fail for the missing 300 days in the train_labels.csv file: from 3/25/2017 to 1/17/2018.

# Workflow:

1. Visualization
    - Quick Exploration
    - Exploratory Visualization
2. Preprocessing
3. Scaling
4. Splitting dataset
5. Implementation of the benchmark model
6. Implementation of the selected model
7. Comparison between the 2 models

# 1. Vizualization:

## 1.1 Quick Exploration & Cleaning:


```python
%matplotlib inline
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
from datetime import datetime                    # manages dates - needed to convert to dates for matplotlib
import seaborn as sns                            # viz

# Retrieve features and labels
features = pd.read_excel('feature.xlsx', skiprows=0, header=1, index_col=0, parse_dates=['date'])
label = pd.read_csv('train_label.csv', header=0, parse_dates=['date'])

# Number of lines read for each file
print('Size of training dataset:', len(features.index))
print('Number of features:', len(features.columns))
print('Size of training labels:', len(label.index))
print('Number of labels:', len(label.columns)-1) # -1 to remove the date
```

    Size of training dataset: 983
    Number of features: 131
    Size of training labels: 684
    Number of labels: 1
    

We can see that the number of labels is 684 and not 683, This is due to the fact that the first label of the labels file is empty. We will clean this for now.

Fist row is removed (date 05/03/2015 - 3rd of May 2015)


```python
label = label.drop([0], axis=0)
print('Size of training labels:', len(label.index))
print('Number of labels:', len(label.columns)-1)
```

    Size of training labels: 683
    Number of labels: 1
    

The problem is that now the two dataframes are not symmetrical: Label starts at date 05/04/2015 - 4th of May 2015 and Features start at date 05/03/2015 - 3rd of May 2015.

In order to have the two dataframes with the same starting date, we also need to remove first row from features dataframe.


```python
features = features.drop([0], axis=0)
print('Size of training features:', len(features.index))
print('Number of features:', len(features.columns))
```

    Size of training features: 982
    Number of features: 131
    


```python
print('labels columns:', label.columns)
print('features columns:', features.columns)
```

    labels columns: Index(['date', 'label'], dtype='object')
    features columns: Index([       'date',     136088194,     136088202,     136088394,
               136088802,     136089546,     136110468,     136216674,
               136222202,     136222210,
           ...
           '136225010.4', '136675426.4', '136675626.4', '136676578.4',
           '136676650.4', '136676666.4', '136676682.4', '136676698.4',
           '136676714.4', '136676754.4'],
          dtype='object', length=131)
    

We can see that the columns names for features are not explicit and repeat. Pandas added a suffix for repeating columns (.1 for 1st repetition, .2 for second repetition ...).

First thing we will do here is rename columns and add explicit suffixes.


```python
# In order to rename columns, we need to first convert columns from int names to string
features.columns = features.columns.astype(str)

# Preparing dictionaries that will be used for renaming columns
real_to_made_errors = {
    '136088194': 'error1', '136088202': 'error2', '136088394': 'error3',
    '136088802': 'error4', '136089546': 'error5', '136110468': 'error6',
    '136216674': 'error7', '136222202': 'error8', '136222210': 'error9',
    '136222234': 'error10', '136222250': 'error11', '136222882': 'error12',
    '136223186': 'error13', '136224578': 'error14', '136224586': 'error15',
    '136224978': 'error16', '136225010': 'error17', '136675426': 'error18',
    '136675626': 'error19', '136676578': 'error20', '136676650': 'error21',
    '136676666': 'error22', '136676682': 'error23', '136676698': 'erro24',
    '136676714': 'error25', '136676754': 'error26'   
}

# Inverse dict in case it is used
made_to_real_errors = {v: k for k, v in real_to_made_errors.items()}

# Suffixes dict
suffixes_dict = {'.1':'.max_tick', '.2':'.min_tick', '.3':'.mean_tick', '.4':'.std_dev_tick'}

# Remaining dicts
real_to_made_errors_max_tick = {k+'.1':v+suffixes_dict['.1'] for k, v in real_to_made_errors.items()}
real_to_made_errors_min_tick = {k+'.2':v+suffixes_dict['.2'] for k, v in real_to_made_errors.items()}
real_to_made_errors_mean_tick = {k+'.3':v+suffixes_dict['.3'] for k, v in real_to_made_errors.items()}
real_to_made_errors_std_dev_tick = {k+'.4':v+suffixes_dict['.4'] for k, v in real_to_made_errors.items()}


def rename_columns(df):
    df.rename(index=str, columns=real_to_made_errors, inplace=True)
    df.rename(index=str, columns=real_to_made_errors_max_tick, inplace=True)
    df.rename(index=str, columns=real_to_made_errors_min_tick, inplace=True)
    df.rename(index=str, columns=real_to_made_errors_mean_tick, inplace=True)
    df.rename(index=str, columns=real_to_made_errors_std_dev_tick, inplace=True)

# Renaming
rename_columns(features)        
```

In order to view the descriptions of columns and statistics, we will split the features in five dataframes:
- features_count
- features_max
- features_min
- features_mean
- features_std_dev

That will allow a cleaner description and vizualization


```python
# Preparing columns
cols_to_keep_for_counts = ['date'] + list(real_to_made_errors.values())
cols_to_keep_for_max = ['date'] + list(real_to_made_errors_max_tick.values())
cols_to_keep_for_min = ['date'] + list(real_to_made_errors_min_tick.values())
cols_to_keep_for_mean = ['date'] + list(real_to_made_errors_mean_tick.values())
cols_to_keep_for_std_dev = ['date'] + list(real_to_made_errors_std_dev_tick.values())

# Preparing 5 dataframes
features_count = features[cols_to_keep_for_counts] 
features_max = features[cols_to_keep_for_max] 
features_min = features[cols_to_keep_for_min] 
features_mean = features[cols_to_keep_for_mean] 
features_std_dev = features[cols_to_keep_for_std_dev] 
```


```python
# Describing count dataframe
features_count.describe(include=[np.number])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1</th>
      <th>error2</th>
      <th>error3</th>
      <th>error4</th>
      <th>error5</th>
      <th>error6</th>
      <th>error7</th>
      <th>error8</th>
      <th>error9</th>
      <th>error10</th>
      <th>...</th>
      <th>error17</th>
      <th>error18</th>
      <th>error19</th>
      <th>error20</th>
      <th>error21</th>
      <th>error22</th>
      <th>error23</th>
      <th>erro24</th>
      <th>error25</th>
      <th>error26</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>...</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.803462</td>
      <td>0.568228</td>
      <td>0.033605</td>
      <td>0.469450</td>
      <td>0.148676</td>
      <td>593.344196</td>
      <td>0.011202</td>
      <td>4.502037</td>
      <td>5.155804</td>
      <td>0.102851</td>
      <td>...</td>
      <td>98.311609</td>
      <td>0.260692</td>
      <td>0.006110</td>
      <td>0.369654</td>
      <td>0.155804</td>
      <td>0.662933</td>
      <td>0.565173</td>
      <td>0.701629</td>
      <td>0.965377</td>
      <td>24.369654</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.708513</td>
      <td>11.685608</td>
      <td>0.930780</td>
      <td>3.042107</td>
      <td>0.582004</td>
      <td>3140.748782</td>
      <td>0.264973</td>
      <td>6.169001</td>
      <td>9.730160</td>
      <td>0.631013</td>
      <td>...</td>
      <td>483.464941</td>
      <td>2.324199</td>
      <td>0.119305</td>
      <td>0.987597</td>
      <td>1.206997</td>
      <td>3.653358</td>
      <td>2.789840</td>
      <td>3.333880</td>
      <td>4.500405</td>
      <td>88.188340</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>89.000000</td>
      <td>364.000000</td>
      <td>29.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>67425.000000</td>
      <td>8.000000</td>
      <td>67.000000</td>
      <td>190.000000</td>
      <td>13.000000</td>
      <td>...</td>
      <td>7801.000000</td>
      <td>48.000000</td>
      <td>3.000000</td>
      <td>17.000000</td>
      <td>22.000000</td>
      <td>57.000000</td>
      <td>36.000000</td>
      <td>44.000000</td>
      <td>62.000000</td>
      <td>941.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
# Describing count dataframe
features_max.describe(include=[np.number])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1.max_tick</th>
      <th>error2.max_tick</th>
      <th>error3.max_tick</th>
      <th>error4.max_tick</th>
      <th>error5.max_tick</th>
      <th>error6.max_tick</th>
      <th>error7.max_tick</th>
      <th>error8.max_tick</th>
      <th>error9.max_tick</th>
      <th>error10.max_tick</th>
      <th>...</th>
      <th>error17.max_tick</th>
      <th>error18.max_tick</th>
      <th>error19.max_tick</th>
      <th>error20.max_tick</th>
      <th>error21.max_tick</th>
      <th>error22.max_tick</th>
      <th>error23.max_tick</th>
      <th>erro24.max_tick</th>
      <th>error25.max_tick</th>
      <th>error26.max_tick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>92.000000</td>
      <td>19.000000</td>
      <td>1.0</td>
      <td>26.000000</td>
      <td>5.000000</td>
      <td>902.000000</td>
      <td>1.0</td>
      <td>253.000000</td>
      <td>331.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>446.000000</td>
      <td>11.000000</td>
      <td>0.0</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>38.000000</td>
      <td>40.000000</td>
      <td>46.000000</td>
      <td>50.00000</td>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60844.695652</td>
      <td>52632.368421</td>
      <td>23388.0</td>
      <td>65166.230769</td>
      <td>61920.200000</td>
      <td>78668.431264</td>
      <td>74017.0</td>
      <td>71111.861660</td>
      <td>69671.383686</td>
      <td>41114.333333</td>
      <td>...</td>
      <td>76852.827354</td>
      <td>57642.272727</td>
      <td>NaN</td>
      <td>60760.714286</td>
      <td>56119.272727</td>
      <td>63019.315789</td>
      <td>62505.950000</td>
      <td>62877.413043</td>
      <td>69828.70000</td>
      <td>79838.325581</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25492.858734</td>
      <td>26965.490554</td>
      <td>NaN</td>
      <td>23197.732697</td>
      <td>22325.682782</td>
      <td>10145.445081</td>
      <td>NaN</td>
      <td>13612.902271</td>
      <td>16840.640244</td>
      <td>29231.910600</td>
      <td>...</td>
      <td>12410.192230</td>
      <td>27291.258385</td>
      <td>NaN</td>
      <td>16939.092299</td>
      <td>16657.302873</td>
      <td>18663.902079</td>
      <td>18759.840344</td>
      <td>21887.960434</td>
      <td>16756.86156</td>
      <td>10878.702815</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1336.000000</td>
      <td>443.000000</td>
      <td>23388.0</td>
      <td>5959.000000</td>
      <td>42144.000000</td>
      <td>3592.000000</td>
      <td>74017.0</td>
      <td>2955.000000</td>
      <td>3935.000000</td>
      <td>21413.000000</td>
      <td>...</td>
      <td>1654.000000</td>
      <td>14400.000000</td>
      <td>NaN</td>
      <td>29883.000000</td>
      <td>26726.000000</td>
      <td>17878.000000</td>
      <td>21194.000000</td>
      <td>4493.000000</td>
      <td>28506.00000</td>
      <td>24925.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48981.750000</td>
      <td>32740.000000</td>
      <td>23388.0</td>
      <td>56993.750000</td>
      <td>44111.000000</td>
      <td>76662.000000</td>
      <td>74017.0</td>
      <td>64558.000000</td>
      <td>63094.000000</td>
      <td>24321.000000</td>
      <td>...</td>
      <td>73080.500000</td>
      <td>32331.500000</td>
      <td>NaN</td>
      <td>53542.500000</td>
      <td>47022.000000</td>
      <td>48862.750000</td>
      <td>50276.500000</td>
      <td>45378.500000</td>
      <td>56424.00000</td>
      <td>79130.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>71915.500000</td>
      <td>60880.000000</td>
      <td>23388.0</td>
      <td>73447.000000</td>
      <td>51145.000000</td>
      <td>82267.000000</td>
      <td>74017.0</td>
      <td>74243.000000</td>
      <td>74867.000000</td>
      <td>27229.000000</td>
      <td>...</td>
      <td>82226.500000</td>
      <td>66159.000000</td>
      <td>NaN</td>
      <td>68564.000000</td>
      <td>56822.000000</td>
      <td>65605.000000</td>
      <td>63834.000000</td>
      <td>69261.500000</td>
      <td>78662.00000</td>
      <td>84115.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>81008.500000</td>
      <td>71303.000000</td>
      <td>23388.0</td>
      <td>83917.500000</td>
      <td>86099.000000</td>
      <td>85017.500000</td>
      <td>74017.0</td>
      <td>81276.000000</td>
      <td>82652.500000</td>
      <td>50965.000000</td>
      <td>...</td>
      <td>85060.500000</td>
      <td>82366.000000</td>
      <td>NaN</td>
      <td>69240.500000</td>
      <td>67940.500000</td>
      <td>78911.500000</td>
      <td>78315.500000</td>
      <td>82275.500000</td>
      <td>83030.00000</td>
      <td>85893.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>86362.000000</td>
      <td>86264.000000</td>
      <td>23388.0</td>
      <td>86380.000000</td>
      <td>86102.000000</td>
      <td>86399.000000</td>
      <td>74017.0</td>
      <td>86390.000000</td>
      <td>86398.000000</td>
      <td>74701.000000</td>
      <td>...</td>
      <td>86399.000000</td>
      <td>85648.000000</td>
      <td>NaN</td>
      <td>81312.000000</td>
      <td>78864.000000</td>
      <td>86342.000000</td>
      <td>86312.000000</td>
      <td>86385.000000</td>
      <td>86337.00000</td>
      <td>86398.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
# Describing count dataframe
features_min.describe(include=[np.number])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1.min_tick</th>
      <th>error2.min_tick</th>
      <th>error3.min_tick</th>
      <th>error4.min_tick</th>
      <th>error5.min_tick</th>
      <th>error6.min_tick</th>
      <th>error7.min_tick</th>
      <th>error8.min_tick</th>
      <th>error9.min_tick</th>
      <th>error10.min_tick</th>
      <th>...</th>
      <th>error17.min_tick</th>
      <th>error18.min_tick</th>
      <th>error19.min_tick</th>
      <th>error20.min_tick</th>
      <th>error21.min_tick</th>
      <th>error22.min_tick</th>
      <th>error23.min_tick</th>
      <th>erro24.min_tick</th>
      <th>error25.min_tick</th>
      <th>error26.min_tick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>92.000000</td>
      <td>19.000000</td>
      <td>1.0</td>
      <td>26.000000</td>
      <td>5.000000</td>
      <td>902.000000</td>
      <td>1.0</td>
      <td>253.000000</td>
      <td>331.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>446.000000</td>
      <td>11.000000</td>
      <td>0.0</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>38.000000</td>
      <td>40.0000</td>
      <td>46.000000</td>
      <td>50.000000</td>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19352.934783</td>
      <td>30668.052632</td>
      <td>17770.0</td>
      <td>11272.115385</td>
      <td>26125.600000</td>
      <td>7414.669623</td>
      <td>72330.0</td>
      <td>13900.158103</td>
      <td>16823.806647</td>
      <td>35599.333333</td>
      <td>...</td>
      <td>9621.769058</td>
      <td>27691.090909</td>
      <td>NaN</td>
      <td>21683.857143</td>
      <td>49714.636364</td>
      <td>21985.552632</td>
      <td>23231.9250</td>
      <td>20797.478261</td>
      <td>19518.060000</td>
      <td>6926.286822</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20406.903018</td>
      <td>23818.831995</td>
      <td>NaN</td>
      <td>11667.044576</td>
      <td>20040.352948</td>
      <td>9252.084697</td>
      <td>NaN</td>
      <td>12646.684779</td>
      <td>17289.067373</td>
      <td>34178.327963</td>
      <td>...</td>
      <td>12701.491205</td>
      <td>27735.217466</td>
      <td>NaN</td>
      <td>22823.526630</td>
      <td>18919.040997</td>
      <td>20983.719040</td>
      <td>21274.7538</td>
      <td>20619.756392</td>
      <td>21910.692444</td>
      <td>12496.345763</td>
    </tr>
    <tr>
      <th>min</th>
      <td>24.000000</td>
      <td>262.000000</td>
      <td>17770.0</td>
      <td>117.000000</td>
      <td>6081.000000</td>
      <td>0.000000</td>
      <td>72330.0</td>
      <td>16.000000</td>
      <td>62.000000</td>
      <td>9893.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>330.000000</td>
      <td>NaN</td>
      <td>1942.000000</td>
      <td>22181.000000</td>
      <td>172.000000</td>
      <td>122.0000</td>
      <td>177.000000</td>
      <td>89.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3109.500000</td>
      <td>10781.000000</td>
      <td>17770.0</td>
      <td>1335.000000</td>
      <td>14246.000000</td>
      <td>1532.250000</td>
      <td>72330.0</td>
      <td>4173.000000</td>
      <td>3340.500000</td>
      <td>16206.000000</td>
      <td>...</td>
      <td>1092.750000</td>
      <td>8696.000000</td>
      <td>NaN</td>
      <td>8078.500000</td>
      <td>31509.500000</td>
      <td>4478.250000</td>
      <td>7370.5000</td>
      <td>5721.750000</td>
      <td>1483.000000</td>
      <td>499.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13020.500000</td>
      <td>28517.000000</td>
      <td>17770.0</td>
      <td>6457.500000</td>
      <td>16360.000000</td>
      <td>4099.000000</td>
      <td>72330.0</td>
      <td>10008.000000</td>
      <td>10709.000000</td>
      <td>22519.000000</td>
      <td>...</td>
      <td>4444.500000</td>
      <td>12526.000000</td>
      <td>NaN</td>
      <td>15185.000000</td>
      <td>54303.000000</td>
      <td>17383.500000</td>
      <td>17229.5000</td>
      <td>14345.000000</td>
      <td>8364.000000</td>
      <td>1776.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>30806.500000</td>
      <td>44380.000000</td>
      <td>17770.0</td>
      <td>19696.500000</td>
      <td>39929.000000</td>
      <td>9710.750000</td>
      <td>72330.0</td>
      <td>20226.000000</td>
      <td>23246.000000</td>
      <td>48452.500000</td>
      <td>...</td>
      <td>13428.500000</td>
      <td>47217.000000</td>
      <td>NaN</td>
      <td>25332.500000</td>
      <td>64030.000000</td>
      <td>33437.000000</td>
      <td>34451.2500</td>
      <td>32864.500000</td>
      <td>33591.000000</td>
      <td>6886.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79203.000000</td>
      <td>83854.000000</td>
      <td>17770.0</td>
      <td>44212.000000</td>
      <td>54012.000000</td>
      <td>70390.000000</td>
      <td>72330.0</td>
      <td>60508.000000</td>
      <td>75746.000000</td>
      <td>74386.000000</td>
      <td>...</td>
      <td>80881.000000</td>
      <td>79428.000000</td>
      <td>NaN</td>
      <td>67838.000000</td>
      <td>78789.000000</td>
      <td>79619.000000</td>
      <td>79330.0000</td>
      <td>79602.000000</td>
      <td>74904.000000</td>
      <td>66685.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
# Describing count dataframe
features_mean.describe(include=[np.number])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1.mean_tick</th>
      <th>error3.mean_tick</th>
      <th>error4.mean_tick</th>
      <th>error5.mean_tick</th>
      <th>error6.mean_tick</th>
      <th>error7.mean_tick</th>
      <th>error8.mean_tick</th>
      <th>error9.mean_tick</th>
      <th>error10.mean_tick</th>
      <th>error11.mean_tick</th>
      <th>...</th>
      <th>error17.mean_tick</th>
      <th>error18.mean_tick</th>
      <th>error19.mean_tick</th>
      <th>error20.mean_tick</th>
      <th>error21.mean_tick</th>
      <th>error22.mean_tick</th>
      <th>error23.mean_tick</th>
      <th>erro24.mean_tick</th>
      <th>error25.mean_tick</th>
      <th>error26.mean_tick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>5.000000</td>
      <td>902.000000</td>
      <td>1.000</td>
      <td>253.000000</td>
      <td>331.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>446.000000</td>
      <td>11.000000</td>
      <td>0.0</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>38.000000</td>
      <td>40.000000</td>
      <td>46.000000</td>
      <td>50.000000</td>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41404.336526</td>
      <td>21390.103448</td>
      <td>36175.135614</td>
      <td>47436.213333</td>
      <td>43549.438347</td>
      <td>73165.375</td>
      <td>43382.712687</td>
      <td>42997.192581</td>
      <td>39324.716026</td>
      <td>61803.833333</td>
      <td>...</td>
      <td>43052.045850</td>
      <td>46777.349116</td>
      <td>NaN</td>
      <td>38541.306543</td>
      <td>54077.413699</td>
      <td>39167.554604</td>
      <td>43992.643489</td>
      <td>40826.362233</td>
      <td>43834.372925</td>
      <td>42793.323113</td>
    </tr>
    <tr>
      <th>std</th>
      <td>21784.543379</td>
      <td>NaN</td>
      <td>15603.199090</td>
      <td>19754.974726</td>
      <td>11563.244543</td>
      <td>NaN</td>
      <td>16126.799794</td>
      <td>16360.804013</td>
      <td>30721.218126</td>
      <td>NaN</td>
      <td>...</td>
      <td>12569.103204</td>
      <td>22391.885922</td>
      <td>NaN</td>
      <td>15539.230834</td>
      <td>16511.678634</td>
      <td>19672.382734</td>
      <td>18206.612388</td>
      <td>19215.872278</td>
      <td>17383.852314</td>
      <td>14195.341603</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1309.600000</td>
      <td>21390.103448</td>
      <td>2925.000000</td>
      <td>23405.166667</td>
      <td>1526.846154</td>
      <td>73165.375</td>
      <td>1419.125000</td>
      <td>2479.300000</td>
      <td>17907.923077</td>
      <td>61803.833333</td>
      <td>...</td>
      <td>1653.048780</td>
      <td>12066.714286</td>
      <td>NaN</td>
      <td>22668.250000</td>
      <td>26701.400000</td>
      <td>11406.600000</td>
      <td>10371.600000</td>
      <td>4298.333333</td>
      <td>12767.789474</td>
      <td>9460.812500</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23488.493405</td>
      <td>21390.103448</td>
      <td>27876.451923</td>
      <td>39119.800000</td>
      <td>36860.965696</td>
      <td>73165.375</td>
      <td>30056.000000</td>
      <td>29741.255482</td>
      <td>21725.274038</td>
      <td>61803.833333</td>
      <td>...</td>
      <td>35192.235294</td>
      <td>27808.687500</td>
      <td>NaN</td>
      <td>27441.676471</td>
      <td>41551.444444</td>
      <td>24217.642857</td>
      <td>28064.531250</td>
      <td>25996.045017</td>
      <td>30957.919355</td>
      <td>32108.367550</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>42962.812500</td>
      <td>21390.103448</td>
      <td>34657.316667</td>
      <td>40870.500000</td>
      <td>44151.912119</td>
      <td>73165.375</td>
      <td>44776.600000</td>
      <td>44137.454545</td>
      <td>25542.625000</td>
      <td>61803.833333</td>
      <td>...</td>
      <td>42887.945233</td>
      <td>50222.020833</td>
      <td>NaN</td>
      <td>37131.000000</td>
      <td>56128.000000</td>
      <td>34458.972222</td>
      <td>44076.192857</td>
      <td>37498.500000</td>
      <td>42060.733333</td>
      <td>41704.854167</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>57259.062500</td>
      <td>21390.103448</td>
      <td>45552.921875</td>
      <td>59337.800000</td>
      <td>50778.058333</td>
      <td>73165.375</td>
      <td>55203.500000</td>
      <td>54446.441667</td>
      <td>50033.112500</td>
      <td>61803.833333</td>
      <td>...</td>
      <td>50780.625000</td>
      <td>61763.138889</td>
      <td>NaN</td>
      <td>43455.071429</td>
      <td>64098.583333</td>
      <td>50510.433198</td>
      <td>54615.860294</td>
      <td>55625.029545</td>
      <td>56199.431235</td>
      <td>51832.861702</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81331.307692</td>
      <td>21390.103448</td>
      <td>72848.272727</td>
      <td>74447.800000</td>
      <td>83102.799940</td>
      <td>73165.375</td>
      <td>83379.142857</td>
      <td>81172.866667</td>
      <td>74523.600000</td>
      <td>61803.833333</td>
      <td>...</td>
      <td>84920.628571</td>
      <td>79497.166667</td>
      <td>NaN</td>
      <td>68196.400000</td>
      <td>78815.600000</td>
      <td>81801.833333</td>
      <td>81529.222222</td>
      <td>81696.000000</td>
      <td>81244.718750</td>
      <td>74623.950495</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>




```python
# Describing count dataframe
features_std_dev.describe(include=[np.number])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1.std_dev_tick</th>
      <th>error2.std_dev_tick</th>
      <th>error3.std_dev_tick</th>
      <th>error4.std_dev_tick</th>
      <th>error5.std_dev_tick</th>
      <th>error6.std_dev_tick</th>
      <th>error7.std_dev_tick</th>
      <th>error8.std_dev_tick</th>
      <th>error9.std_dev_tick</th>
      <th>error10.std_dev_tick</th>
      <th>...</th>
      <th>error17.std_dev_tick</th>
      <th>error18.std_dev_tick</th>
      <th>error19.std_dev_tick</th>
      <th>error20.std_dev_tick</th>
      <th>error21.std_dev_tick</th>
      <th>error22.std_dev_tick</th>
      <th>error23.std_dev_tick</th>
      <th>erro24.std_dev_tick</th>
      <th>error25.std_dev_tick</th>
      <th>error26.std_dev_tick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>92.000000</td>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>5.000000</td>
      <td>902.000000</td>
      <td>1.000000</td>
      <td>253.000000</td>
      <td>331.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>446.000000</td>
      <td>11.000000</td>
      <td>0.0</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>38.000000</td>
      <td>40.000000</td>
      <td>46.000000</td>
      <td>50.000000</td>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14563.788427</td>
      <td>7920.794347</td>
      <td>1877.496953</td>
      <td>18305.520687</td>
      <td>15705.164811</td>
      <td>22758.667047</td>
      <td>597.602576</td>
      <td>20710.841884</td>
      <td>19168.115749</td>
      <td>1859.185209</td>
      <td>...</td>
      <td>21528.991962</td>
      <td>9276.242498</td>
      <td>NaN</td>
      <td>14298.442241</td>
      <td>2861.203915</td>
      <td>13988.284805</td>
      <td>14244.256825</td>
      <td>14332.896055</td>
      <td>17975.277266</td>
      <td>19609.738290</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10372.395093</td>
      <td>9237.364996</td>
      <td>NaN</td>
      <td>8510.338381</td>
      <td>11455.283091</td>
      <td>5648.236456</td>
      <td>NaN</td>
      <td>7121.437626</td>
      <td>8812.417060</td>
      <td>1769.840257</td>
      <td>...</td>
      <td>6337.472007</td>
      <td>9854.359214</td>
      <td>NaN</td>
      <td>11431.755718</td>
      <td>5170.563133</td>
      <td>8626.450595</td>
      <td>8783.676041</td>
      <td>8625.009447</td>
      <td>10803.395306</td>
      <td>6791.399411</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.023769</td>
      <td>71.909434</td>
      <td>1877.496953</td>
      <td>14.293355</td>
      <td>1009.344887</td>
      <td>6.833333</td>
      <td>597.602576</td>
      <td>1050.176254</td>
      <td>8.246211</td>
      <td>118.280599</td>
      <td>...</td>
      <td>0.705380</td>
      <td>38.405295</td>
      <td>NaN</td>
      <td>425.067406</td>
      <td>20.983327</td>
      <td>1261.217798</td>
      <td>865.889170</td>
      <td>149.092812</td>
      <td>88.538833</td>
      <td>717.884695</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6534.539960</td>
      <td>270.622405</td>
      <td>1877.496953</td>
      <td>13944.264941</td>
      <td>10713.738534</td>
      <td>19582.074586</td>
      <td>597.602576</td>
      <td>16357.294657</td>
      <td>13078.501647</td>
      <td>960.474189</td>
      <td>...</td>
      <td>17858.809822</td>
      <td>625.503204</td>
      <td>NaN</td>
      <td>7649.375712</td>
      <td>49.797046</td>
      <td>5861.044529</td>
      <td>7853.859023</td>
      <td>7145.035479</td>
      <td>9100.182210</td>
      <td>15139.849784</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14295.741195</td>
      <td>4356.033000</td>
      <td>1877.496953</td>
      <td>18465.534889</td>
      <td>15694.077600</td>
      <td>23403.535493</td>
      <td>597.602576</td>
      <td>21259.412238</td>
      <td>20203.628325</td>
      <td>1802.667780</td>
      <td>...</td>
      <td>22863.378846</td>
      <td>5092.739511</td>
      <td>NaN</td>
      <td>12447.336848</td>
      <td>167.069185</td>
      <td>13959.512501</td>
      <td>14912.032919</td>
      <td>13919.803616</td>
      <td>17544.791642</td>
      <td>20655.670931</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20955.026855</td>
      <td>12894.259967</td>
      <td>1877.496953</td>
      <td>24416.129554</td>
      <td>18846.116754</td>
      <td>26416.260063</td>
      <td>597.602576</td>
      <td>25565.145433</td>
      <td>25606.680951</td>
      <td>2729.637515</td>
      <td>...</td>
      <td>25503.957648</td>
      <td>16844.472453</td>
      <td>NaN</td>
      <td>18455.789345</td>
      <td>2228.148483</td>
      <td>20846.103258</td>
      <td>21075.838193</td>
      <td>22159.792132</td>
      <td>26862.753264</td>
      <td>24271.551851</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41813.563785</td>
      <td>28253.045521</td>
      <td>1877.496953</td>
      <td>31576.030745</td>
      <td>32262.546284</td>
      <td>38852.530722</td>
      <td>597.602576</td>
      <td>36759.709097</td>
      <td>44463.169582</td>
      <td>3656.607249</td>
      <td>...</td>
      <td>40498.274760</td>
      <td>26490.226429</td>
      <td>NaN</td>
      <td>35006.361316</td>
      <td>15250.921875</td>
      <td>31232.119085</td>
      <td>35097.775546</td>
      <td>29600.736779</td>
      <td>44234.371299</td>
      <td>33979.509589</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
# Check total number of errors
print("Total number of errors in dataset is ", label['label'].sum())
print("Percentage of errors in dataset is {:.2f}%".format((label['label'].sum()/label['label'].count())*100))
```

    Total number of errors in dataset is  44.0
    Percentage of errors in dataset is 6.44%
    

## 1.2 Exploratory Visualization

Visualization was done without scaling bu the results were useless: 
- Different scales for each error stat
- Some error counts were simply undistinguishable from the other

That makes rescaling necessary to compare between the different error stats.

Besides, in order to visualize clearly we choose to focus only on errors count. And to have at most 3 error counts displayed in a graph.

The error counts will give us information on trends and seasonality. Indeed, the other stats (max, min, mean and std_dev) only give us details for the corresponding day but not through time.

We could see the error count as a macro indicator and the other stats (max, min, mean and std_dev) as micro stats.


```python
sns.set()

def rescale_element(x, min, max):
    return ((x - min)/(max - min))

def get_failure_indexes(label_df):
    """Returns list of indexes of days where equipment fails"""
    return label.index[label['label'] == 1].tolist()

def get_failure_only_label_dataframe(label_df):
    """Returns a dataframe containing only the equipment failures"""
    return label_df.loc[get_failure_indexes,:]

def rescale_column(df, col):
    """Scales values between 0 and 1"""
    min = df[col].min()
    max = df[col].max()
    return df[col].apply(rescale_element, args=(min,max))

def plot_dataframe(features_df, label_df, start, end, title, grid):
    """ Prints all 26 error types time series 
    and adds markers at dates where the equipment fails.
    start: must be greater or equal to 1 (index 1 is the date)
    end: must be less or equal to 27"""
    
    plt.figure(figsize=(15, 7))
    
    # Plots errors stats
    for col in list(features_df.columns)[start:end]:
        plt.plot(features_df.date.astype(datetime), rescale_column(features_df, col), label=col)
        
    # Plot equipment failures as big red points at y=0
    plt.plot(get_failure_only_label_dataframe(label_df).date.astype(datetime), 
             pd.Series(np.zeros(len(get_failure_indexes(label_df)))),#get_failure_only_label_dataframe(label_df).label-1, 
             "ko", markersize=5, label="failure")
    
    # Titles, Legends and formatting
    plt.title(title)
    plt.grid(grid)
    plt.legend(ncol=3, loc='upper left')
    plt.show()
    

# Plot all errors (4 error counts in same graph)
plot_dataframe(features_count, label, 1, 4, "Errors Count 1 to 3", True)
plot_dataframe(features_count, label, 4, 8, "Errors Count 4 to 7", True)
plot_dataframe(features_count, label, 8, 12, "Errors Count 8 to 11", True)
plot_dataframe(features_count, label, 12, 16, "Errors Count 12 to 15", True)
plot_dataframe(features_count, label, 16, 20, "Errors Count 16 to 19", True)
plot_dataframe(features_count, label, 20, 24, "Errors Count 20 to 23", True)
plot_dataframe(features_count, label, 24, 27, "Errors Count 24 to 26", True)
```


![png](capstone_project_files/capstone_project_23_0.png)



![png](capstone_project_files/capstone_project_23_1.png)



![png](capstone_project_files/capstone_project_23_2.png)



![png](capstone_project_files/capstone_project_23_3.png)



![png](capstone_project_files/capstone_project_23_4.png)



![png](capstone_project_files/capstone_project_23_5.png)



![png](capstone_project_files/capstone_project_23_6.png)



```python
def plot_figures_grid(features_df, label_df, ncols, nrows):
    """Plot each error in a separate graph"""

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=True)
    fig.set_size_inches((15, 30))
    for ind, col in enumerate(list(features_df.columns)[1:]):
        axeslist.ravel()[ind].plot(features_df.date.astype(datetime), rescale_column(features_df, col), label=col)
        axeslist.ravel()[ind].plot(get_failure_only_label_dataframe(label_df).date.astype(datetime), 
                 pd.Series(np.zeros(len(get_failure_indexes(label_df)))), 
                 "ko", markersize=2, label="failure")
        axeslist.ravel()[ind].set_title(col)
        axeslist.ravel()[ind].set_axis_off()
    #plt.tight_layout() # optional

# Plot figures in a grid
plot_figures_grid(features_count, label, 2, 13)
```


![png](capstone_project_files/capstone_project_24_0.png)


### Visualization results

From all the plots above we can notice three types of error_counts:
- Peaks: contains one or at most two peaks of errors throughout the time period (errors: 2, 3, 7, 19)
- Short-lived: has activvity during a limited amount of time - weeks or at most 6 months (errors: 6, 17, 22, 23, 24, 25, 26)
- Constant Activity: has activity throughout all the time period (errors: 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21)

It appears that the failures are not directly linked to a specific error count (all failures are not happening when we have peaks at error21 for example). 

However we can clearly see that some error counts peaks coincide with some failures (for example error5 and error8 seem to be correlated to the failures) so we can conclude that each failure is closely linked with a combination of error counts.

For trends we can see that the errors don't follow specific trends (increasing or decreasing over time).

For seasonality, apart from errors 5, 11 and 13 all error counts seem to be random. Further investigation could be done on that area but due to the submission deadline (today) I can unfortunately not go deeper for now.

# 2. Preprocessing

Definition of **stationarity**: 

**Stationarity** is one of the most important concepts when working with time series data. A stationary series is one in which the properties – mean, variance and covariance, do not vary with time.

Let us understand this using an intuitive example. Consider the three plots shown below:
<img src="stationary_examples.png">

In the first plot, we can clearly see that the mean varies (increases) with time which results in an upward trend. Thus, this is a non-stationary series. For a series to be classified as stationary, it should not exhibit a trend.
Moving on to the second plot, we certainly do not see a trend in the series, but the variance of the series is a function of time. As mentioned previously, a stationary series must have a constant variance.
If you look at the third plot, the spread becomes closer as the time increases, which implies that the covariance is a function of time.

The three examples shown above represent non-stationary time series. Now let's look at another plot:
<img src="stationary_example.png">

In this case, the mean, variance and covariance are constant with time. This is what a stationary time series looks like.

Predicting future values using the latter plot would be easier. Indeed, most statistical models require the series to be stationary to make effective and precise predictions.

**Analyzing Correlations for error counts only**

The ticks represent the first, last, mean, stddev of the time of the day whem the errors occured. This data is repeating every day so we clearly expect a high correlation here.

For errors count, we need to see if they are correlated and we will apply a specific test for that: The Coint Johansen test. 

According to this [source](https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/), if this test returns values less than 1 in modulus, the features can be considered stationary.

We will apply this test for a lag of 1 as it was done in the source. This considers the stationarity with a lag of 1 day.

As the test only accepts 12 values for each call we will split the data in two subsets (columns 1 to 12 and columns 13 to 24). We will skip the remaining columns (25 and 26) for simplicity.


```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen
colums_1_to_12 = list(features_count.columns)[1:13]
colums_13_to_24 = list(features_count.columns)[14:25]
features_count_first_half = features_count[colums_1_to_12]
features_count_second_half = features_count[colums_13_to_24]
```


```python
coint_johansen(features_count_first_half,-1,1).eig
```




    array([0.41020122, 0.34994062, 0.33663433, 0.33100519, 0.3076108 ,
           0.28157895, 0.27306417, 0.23438865, 0.2140128 , 0.19591291,
           0.15592227, 0.09700534])




```python
coint_johansen(features_count_second_half,-1,1).eig
```




    array([0.38855547, 0.34959398, 0.32255814, 0.31596682, 0.30195993,
           0.27833122, 0.24824906, 0.20510221, 0.18712989, 0.12083628,
           0.10222243])



The values returned here are all less than 1 in modulus. **The multivariate time series can be considered stationary**

**Some simple preprocessing is also needed here:**
- **We will also remove error19 column in all tick dataframes as all its values are NaN/0**
- **We will only fill missing/non numerical values with 0 for all ticks**
- **We will also change the dataframe indexes to have dates instead of 1 to 982**


```python
# Remove error19 columns from all tick dataframes
features_max.drop(['error19.max_tick'], axis=1, inplace=True)
features_min.drop(['error19.min_tick'], axis=1, inplace=True)
features_mean.drop(['error19.mean_tick'], axis=1, inplace=True)
features_std_dev.drop(['error19.std_dev_tick'], axis=1, inplace=True)

# Convert all values to numeric types
features_count = features_count.apply(pd.to_numeric, errors='coerce')
features_max = features_max.apply(pd.to_numeric, errors='coerce')
features_min = features_min.apply(pd.to_numeric, errors='coerce')
features_mean = features_mean.apply(pd.to_numeric, errors='coerce')
features_std_dev = features_std_dev.apply(pd.to_numeric, errors='coerce')

# Filling missing values with 0
features_count = features_count.fillna(0)
features_max = features_max.fillna(0)
features_min = features_min.fillna(0) 
features_mean = features_mean.fillna(0)
features_std_dev = features_std_dev.fillna(0)


def replace_index_with_dates(df):
    """
        Too many problems with different date formats for labels and 
        features.
        This function makes sure that the index is the same date for features and
        labels.
    """
    df.set_index(pd.date_range(start='5/4/2015', periods=len(df.index)), inplace=True)
    df.drop(['date'], axis=1, inplace=True)
    

# Replacing index with date   
replace_index_with_dates(features_count)
replace_index_with_dates(features_max)
replace_index_with_dates(features_min)
replace_index_with_dates(features_mean)
replace_index_with_dates(features_std_dev)
replace_index_with_dates(label)
```

    c:\users\horki\miniconda3\envs\capstone_project\lib\site-packages\pandas\core\frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)
    

**TO NOTE**: 

Stationarity is mostly needed for our benchmark model (VAR) and not for our selected model. Neural Networks do not require stationarity.

"Neural networks excel at capturing complex relationships between features (in this case historical data), therefore neither trend nor seasonality removal was performed." -> see [article](https://github.com/shellshock1911/Sky-Cast-Capstone/blob/master/final_report.pdf)

# 3. Scaling

A simple scaler will be chosen here: **MinMax Scaler** from scikit-learn. It scales the values between 0 and 1.


```python
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Standard Scaler
error_count_scaler = MinMaxScaler()

# All columns but date
features_count_all_columns = list(features_count.columns)

# Apply Standard Scaler to errors count
features_count_scaled_nparray = error_count_scaler.fit_transform(features_count)
features_count_scaled = pd.DataFrame(features_count_scaled_nparray, index=features_count.index, columns=features_count.columns)

features_count_scaled.describe(include='all')
```

    c:\users\horki\miniconda3\envs\capstone_project\lib\site-packages\sklearn\preprocessing\data.py:323: DataConversionWarning: Data with input dtype int64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1</th>
      <th>error2</th>
      <th>error3</th>
      <th>error4</th>
      <th>error5</th>
      <th>error6</th>
      <th>error7</th>
      <th>error8</th>
      <th>error9</th>
      <th>error10</th>
      <th>...</th>
      <th>error17</th>
      <th>error18</th>
      <th>error19</th>
      <th>error20</th>
      <th>error21</th>
      <th>error22</th>
      <th>error23</th>
      <th>erro24</th>
      <th>error25</th>
      <th>error26</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>...</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.020264</td>
      <td>0.001561</td>
      <td>0.001159</td>
      <td>0.011736</td>
      <td>0.024779</td>
      <td>0.008800</td>
      <td>0.001400</td>
      <td>0.067195</td>
      <td>0.027136</td>
      <td>0.007912</td>
      <td>...</td>
      <td>0.012602</td>
      <td>0.005431</td>
      <td>0.002037</td>
      <td>0.021744</td>
      <td>0.007082</td>
      <td>0.011630</td>
      <td>0.015699</td>
      <td>0.015946</td>
      <td>0.015571</td>
      <td>0.025898</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.075377</td>
      <td>0.032103</td>
      <td>0.032096</td>
      <td>0.076053</td>
      <td>0.097001</td>
      <td>0.046581</td>
      <td>0.033122</td>
      <td>0.092075</td>
      <td>0.051211</td>
      <td>0.048539</td>
      <td>...</td>
      <td>0.061975</td>
      <td>0.048421</td>
      <td>0.039768</td>
      <td>0.058094</td>
      <td>0.054864</td>
      <td>0.064094</td>
      <td>0.077496</td>
      <td>0.075770</td>
      <td>0.072587</td>
      <td>0.093718</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000163</td>
      <td>0.000000</td>
      <td>0.029851</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000128</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000311</td>
      <td>0.000000</td>
      <td>0.044776</td>
      <td>0.010526</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000385</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.011236</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000667</td>
      <td>0.000000</td>
      <td>0.074627</td>
      <td>0.036842</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.001538</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
# Standard Scaler
error_ticks_scaler = MinMaxScaler()

# Create a dataframe with all ticks stats (concatenate max_tick, min_tick, mean_tick, std_dev_tick)
all_ticks = pd.concat([features_max, features_min, features_mean, features_std_dev], axis=1)

# All columns but date
all_ticks_all_columns = list(all_ticks.columns)

# Apply Standard Scaler to errors count
all_ticks_scaled_nparray = error_ticks_scaler.fit_transform(all_ticks)
all_ticks_scaled = pd.DataFrame(all_ticks_scaled_nparray, index=all_ticks.index, columns=all_ticks.columns)

all_ticks_scaled.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>error1.max_tick</th>
      <th>error2.max_tick</th>
      <th>error3.max_tick</th>
      <th>error4.max_tick</th>
      <th>error5.max_tick</th>
      <th>error6.max_tick</th>
      <th>error7.max_tick</th>
      <th>error8.max_tick</th>
      <th>error9.max_tick</th>
      <th>error10.max_tick</th>
      <th>...</th>
      <th>error16.std_dev_tick</th>
      <th>error17.std_dev_tick</th>
      <th>error18.std_dev_tick</th>
      <th>error20.std_dev_tick</th>
      <th>error21.std_dev_tick</th>
      <th>error22.std_dev_tick</th>
      <th>error23.std_dev_tick</th>
      <th>erro24.std_dev_tick</th>
      <th>error25.std_dev_tick</th>
      <th>error26.std_dev_tick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>...</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
      <td>982.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.066005</td>
      <td>0.011805</td>
      <td>0.001018</td>
      <td>0.019974</td>
      <td>0.003662</td>
      <td>0.836348</td>
      <td>0.001018</td>
      <td>0.212074</td>
      <td>0.271811</td>
      <td>0.001681</td>
      <td>...</td>
      <td>0.022707</td>
      <td>0.241441</td>
      <td>0.003923</td>
      <td>0.002912</td>
      <td>0.002102</td>
      <td>0.017331</td>
      <td>0.016531</td>
      <td>0.022682</td>
      <td>0.020691</td>
      <td>0.075811</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.224213</td>
      <td>0.094145</td>
      <td>0.031911</td>
      <td>0.128541</td>
      <td>0.053821</td>
      <td>0.273433</td>
      <td>0.031911</td>
      <td>0.368922</td>
      <td>0.397789</td>
      <td>0.035153</td>
      <td>...</td>
      <td>0.110075</td>
      <td>0.285021</td>
      <td>0.052633</td>
      <td>0.042828</td>
      <td>0.039521</td>
      <td>0.101720</td>
      <td>0.094511</td>
      <td>0.119889</td>
      <td>0.104725</td>
      <td>0.207977</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.851827</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.943935</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.981638</td>
      <td>0.000000</td>
      <td>0.465531</td>
      <td>0.735792</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.540393</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 100 columns</p>
</div>



Now preparing dataset before splitting/implementation:
- Putting back errors count and ticks in the same dataframe
- Removing extra lines from the resulting dataframe (lines not having labels - 982 feature lines vs 683 label lines)


```python
# Now putting all features in the same dataframe
all_features_scaled = pd.concat([features_count_scaled, all_ticks_scaled], axis=1)
print("Features Dataframe Infos: ")
all_features_scaled.info()

# Checking labels
print("\nLabel Dataframe Infos: ")
label.info()

# 982 entries in features and 683 entries in labels

# We need to remove the last 299 rows of features
all_features_scaled.drop(all_features_scaled.tail(299).index,inplace=True)

# Checking features again
print("\nFeatures Dataframe Infos: ")
all_features_scaled.info()
```

    Features Dataframe Infos: 
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 982 entries, 2015-05-04 to 2018-01-09
    Freq: D
    Columns: 126 entries, error1 to error26.std_dev_tick
    dtypes: float64(126)
    memory usage: 1014.3 KB
    
    Label Dataframe Infos: 
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 683 entries, 2015-05-04 to 2017-03-16
    Freq: D
    Data columns (total 1 columns):
    label    683 non-null float64
    dtypes: float64(1)
    memory usage: 10.7 KB
    
    Features Dataframe Infos: 
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 683 entries, 2015-05-04 to 2017-03-16
    Freq: D
    Columns: 126 entries, error1 to error26.std_dev_tick
    dtypes: float64(126)
    memory usage: 677.7 KB
    

# 4. Splitting datasets


```python
def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test
```


```python
X_train, X_test, y_train, y_test = timeseries_train_test_split(all_features_scaled, label, test_size=0.2)
```

We also need to join the features and labels in the same dataframe as they will be forecast together. We will not treat this problem as classification problem where we use the features to predict equipment failure.

The features and the label will be forecast using the following two algorithms. We will then calculate the AUC metric.


```python
dataset_train = pd.concat([X_train, y_train], axis=1)
dataset_test = pd.concat([X_test, y_test], axis=1)

print(dataset_train.info())
print(dataset_test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 546 entries, 2015-05-04 to 2016-10-30
    Freq: D
    Columns: 127 entries, error1 to label
    dtypes: float64(127)
    memory usage: 566.0 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 137 entries, 2016-10-31 to 2017-03-16
    Freq: D
    Columns: 127 entries, error1 to label
    dtypes: float64(127)
    memory usage: 142.0 KB
    None
    

Result of splitting:
- **Training dataset length is 546**
- **Testing dataset length is 137**

These results (especially the testing dataset length) will be used for the VAR prediction.
For the VAR prediction the dataset will not be split but the whole dataset will be used to create a predicion dataset. More details will be given below.


```python
dataset_all_with_label = pd.concat([all_features_scaled, label], axis=1)
```

# 5. Implementation of the benchmark model

The benchmark model is the VAR algorithm. It is not in sklearn but in another python library called statsmodel.

While trying to use VAR with the above resulting datasets, we noticed that the VAR cannot work with constant columns. Although there were no constant columns before splitting the datasets to train & test. We now have constant columns in the training dataset.

We will then remove all constant columns to allow the VAR to apply to the data. This operation is specific to the  VAR algorithm, that is why it was not done above during the splitting (Step 4.)


```python
# Remove columns that are constant
dataset_all_with_label_var = dataset_all_with_label.loc[:, (dataset_all_with_label != dataset_all_with_label.iloc[0]).any()]

removed_columns = list()
for col in list(dataset_all_with_label.columns):
    if col not in (list(dataset_all_with_label_var.columns)):
        removed_columns.append(col)
        
print("removed columns : ", removed_columns)
```

    removed columns :  ['error19', 'error3.max_tick', 'error11.max_tick', 'error3.min_tick', 'error11.min_tick', 'error3.mean_tick', 'error11.mean_tick', 'error3.std_dev_tick', 'error11.std_dev_tick']
    


```python
# Remaining features:
print(len(list(dataset_all_with_label_var.columns)))
```

    118
    

The var algorithm expects many parameters but we focused on the following that we noted as the most important:
- The training dataset is needed at the VAR class instanciation
- Then the following 2 params during the algorithm fit call:
    - **maxlags**: the maximum number of steps (in the past) to take into account for the calculation ot the forecast/prediction. Exemple: 10 - take the last 10 days/steps values to do the prediction
    - **trend**: The type of the last element of the equation to calculate the forecast. The VAR equation uses the past elements of all features to calculate the next features and it adds an extra parameter (either a constant or a trend or both). Possible values for this parameter are: constant, constant and trend, constant linear and quadratic trend and eventually no constant. 
- At last the number of steps to predict is needed for the forecast call.  Example: 30 - forecast the next 30 days/steps.

**IMPORTANT**:

We tried to use the VAR algorithm to predict all the testing set at once (one forecast call) and get the AUC but the results were below 40%. The resulting label prediction was a few peaks for the first 3 days followed be a slowly decreasing line that always remained below 0.05. The prediction was 0 for all the testing data.

To improve these results we decided to use the VAR algorithm to predict only one day. The algorithm would be good for predicting the next day only. And to evaluate this algorithm with regards to the testing dataset, the idea was to generate a prediction dataset and calculate the AUC only for the predicted steps.

Here we will not split the training set and the testing set. All the data will be used by the VAR to predict the next day. The predicted days will correspond to the dates in the testing dataset. The testing dataset is indeed the last 137 days of the whole dataset. So we will predict the last 137 days (prediction dataset) and calculate the AUC against the real last 137 days (last 137 rows of the whole dataset).


```python
from statsmodels.tsa.vector_ar.var_model import VAR

dataset_test_length = len(dataset_test.index) #size of the testing sets

#Hyperparameters
maximum_lag = 250 #number of previous steps taken into considerations for predictions
steps_to_predict = 137
trend = 'nc' # “c” - add constant “ct” - constant and trend “ctt” - constant, linear and quadratic trend “nc” - no constant
```


```python
def var_using_one_day_prediction(all_features_scaled_df, maximum_lag, steps_to_predict, trend):
    """
        Forecast given number of steps (steps_to_predict) by using VAR algorithm for 1 day prediction.
        Predicting 1 day and doing it for the given number of steps. Each prediction is using all dataset before the actual 
        predicted day.
    """    
    #Removing steps_to_predict rows from dataframe
    len_features_df = len(all_features_scaled_df.index)
    result_df = all_features_scaled_df.head(len_features_df - steps_to_predict) 
    
    #Repeating one day prediction for all steps to predict
    for step in range(steps_to_predict):
        #Preparing Training dataframe
        current_size = len_features_df - steps_to_predict + step
        temp_df = all_features_scaled_df.head(len_features_df - steps_to_predict + step)  
        #Training
        model_var = VAR(temp_df)
        model_var_fit = model_var.fit(maxlags=maximum_lag, trend=trend)
        #Forecast (1 time step)
        prediction_var_array = model_var_fit.forecast(model_var_fit.y, 1)
        #Create Dataframe of prediction
        prediction_var_df = pd.DataFrame(prediction_var_array,
                            index=pd.date_range(start=all_features_scaled_df.index[len(temp_df.index)], periods=1, freq='D'),
                            columns=all_features_scaled_df.columns)
        #Append prediciton to result dataframe
        result_df = pd.concat([result_df, prediction_var_df], axis=0, ignore_index=False)
        result_df.index.freq=result_df.index.inferred_freq #setting frequency as Daily (inferred freq is daily)

    return result_df

```


```python
prediction_var_df = var_using_one_day_prediction(dataset_all_with_label_var, maximum_lag, steps_to_predict, trend)
```

Features and label are all in the resulting dataframe.

We will now need to split them and check the AUC for the label.


```python
# Removed columns
var_removed_count_columns = ['error19']
var_removed_max_columns = ['error3.max_tick', 'error11.max_tick']
var_removed_min_columns = ['error3.min_tick', 'error11.min_tick']
var_removed_mean_columns = ['error3.mean_tick', 'error11.mean_tick']
var_removed_std_dev_columns = ['error3.std_dev_tick', 'error11.std_dev_tick']

# Recreation of each feature dataframe (column in initial dataframe but not in removed columns)
feature_count_var_columns = [col for col in list(features_count.columns) if col not in var_removed_count_columns]
feature_max_var_columns = [col for col in list(features_max.columns) if col not in var_removed_max_columns]
feature_min_var_columns = [col for col in list(features_min.columns) if col not in var_removed_min_columns]
feature_mean_var_columns = [col for col in list(features_mean.columns) if col not in var_removed_mean_columns]
feature_std_dev_var_columns = [col for col in list(features_std_dev.columns) if col not in var_removed_std_dev_columns]

# No need to recreate the label column as there is only one label column: 'label'

# Split error counts from ticks and label
feature_count_var_scaled = prediction_var_df[feature_count_var_columns]
feature_max_var_scaled = prediction_var_df[feature_max_var_columns]
feature_min_var_scaled = prediction_var_df[feature_min_var_columns]
feature_mean_var_scaled = prediction_var_df[feature_mean_var_columns]
feature_std_dev_var_scaled = prediction_var_df[feature_std_dev_var_columns]
label_var = prediction_var_df['label'] # label was not scaled
```


```python
# Plot predicted and actual
plt.figure(figsize=(15, 7))
plt.plot(feature_count_var_scaled['error8'].tail(steps_to_predict), label="predicted error8") 
plt.plot(all_features_scaled['error8'].tail(steps_to_predict), label="actual error8")

# Titles, Legends and formatting
plt.title("predicted error8 vs actual error8")
plt.grid(True)
plt.legend(ncol=2, loc='upper left')
plt.show()
```


![png](capstone_project_files/capstone_project_60_0.png)



```python
# Plot predicted and actual
plt.figure(figsize=(15, 7))
plt.plot(label_var.tail(steps_to_predict), label="predicted label") 
plt.plot(label.tail(steps_to_predict), label="actual label")

# Titles, Legends and formatting
plt.title("predicted label vs actual")
plt.grid(True)
plt.legend(ncol=2, loc='upper left')
plt.show()
```


![png](capstone_project_files/capstone_project_61_0.png)


## AUC for VAR algorithm


```python
from sklearn.metrics import roc_auc_score

actual_label = label.tail(steps_to_predict)
predicted_label = label_var.tail(steps_to_predict)

# Calculate auc
roc_auc_score(actual_label, predicted_label) 
```




    0.5967741935483871



### The ROC score for the VAR algorithm is at 0.597. It is slightty better than random guess.

From the figures above we can see that the prediction seems better for the errors than for the label. The error prediction is clearly following the same trend as the error it tries to predict.

For the label (equipment failure), it is slightly more difficult to predict and the trend is not as clear as for the error.

The VAR algorithm is a regression algorithm. Each signal is predicted according to a past version of itself and all the other signals. 

This algorithm is then solving this problem using a regression mechanism. This would be fine to predict all error signals as they are all numerical values that change through time. But the label is actually a category (equipment OK or not). Categories cannot be predicted in an optimal manner using a regression algorithm. That is, to my opinion, the limit of the VAR algorithm for this dataset.

# 6. Implementation of the selected model

The selected model is the RNN-Seq2Seq algorithm. It is not in sklearn but we will use another library called Keras for Deep Learning.

We could have chosen a simple many to one architecture to directly compare with the previous benchmark implementation. Many to one architecture consists in using many inputs to predict only one (As we are doing using the VAR algorithm).

Instead, we choose to have a many-to-many implementation (Seq2Seq) to take advantage of the power of RNNs to predict many successive values using many inputs. The number of inputs and predictions is, of course, independent from one another.

<img src="seq2seq.jpg">

## RNN Seq2Seq Recap

RNNs consists of two layers:
- Encoder RNN: Process input sequence and returns it is internal state and its outputs (ignored here)
- Decoder RNN: Predicts next outputs given the encoder state (See architecture below)

<img src="encoder-decoder.png">

In order to use the RNN, we need to be able to feed it with data. Our objective is to use the same data that was used in the benchmark algorithm (all error counts + max ticks + min ticks + mean ticks + std_dev).

The Algorithm will figure out the link between all these elements and hopefully find a way to predict equipment failure one day (or more) in advance.

## Prepare RNN Inputs

Each RNN cell will receive all the data at once so we need to prepare the data. Each RNN cell input will be a concatenation of all errors + ticks + min + max + mean + std_dev.

The Decoder Cell output will be a label.

We already have all the features & labels in pandas dataframes but we will need to convert them in numpy arrays.


```python
# Converting features and label to numpy array with type np.float32 (type needed to be able to train on GPUs)
X_train_np = X_train.values.astype(np.float32) 
y_train_np = y_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)
print(X_train_np.shape)
print(y_train_np.shape)
print(X_test_np.shape)
print(y_test_np.shape)
```

    (546, 126)
    (546, 1)
    (137, 126)
    (137, 1)
    

Now that the data has the right format, we need to create batches for the algorithm to train.

Our dataset is very small, so we will apply a trick here which is generating infinite batches from a finite small dataset (data augmentation).

The trick will consist in a generating function that will provide random batches of data from the existing dataset. Everytime the function is called, it will generate a batch by randomly choosing an offset from which the batch will start. Thus creating infinite batches from a finite dataset.

This will be used to train the algorithm because Deep Learning needs many batches to train.

### Generator Function


```python
import random

def batch_generator(features_array, 
                    label_array, 
                    steps_per_epoch = 100,
                    batch_size = 50, 
                    input_sequence_length = 5, 
                    output_sequence_length = 1):
    """ 
    Generates batches from the fixed elements of the dataset (features_array & label_array).
    Called at each step by the RNN algorithm. Will run with a seed in order to generate same sequence for every epoch.
    Yields a tuple containing:
    - ([encoder_inputs, decoder_inputs], decoder_outputs)
        encoder_inputs shape: (batch_size, input_sequence_length, 126) - set up from features_array data(shape: (batch_size, 126)
        decoder_inputs shape: (batch_size, output_sequence_length, 1) - all values are set 0 (decoder inputs not used here)
        decoder_outputs shape (batch_size, output_sequence_length, 1) - set up from label_array data(shape: (batch_size, 1))
    """
    # Data taken at each step for features (batch_size*input_sequence_length) 
    # should be at most half of actual dataset size (features_array.shape[0]/2) to avoid having similar data for each step
    # We decided to have at most half of the dataset returned at each call
    if not list(features_array) or not list(label_array):
        raise Exception('Empty array')
    elif batch_size <= 0:
        raise Exception('Batch_size cannot be negative nor 0')
    elif steps_per_epoch <= 0:
        raise Exception('Steps_per_epoch cannot be negative nor 0')
    elif input_sequence_length <= 0 or output_sequence_length <= 0:
        raise Exception('Input length cannot be negative or 0')
    elif (features_array.shape[0]/2) < (batch_size*input_sequence_length):
        raise Exception('Too many features taken at each step, reduce batch_size and/or input_input_sequence_length')
    elif input_sequence_length < output_sequence_length :
        raise Exception('This generator does not cover cases when input sequence is greater than output sequence')
    else:
        while True:
            # Reset seed to obtain same sequences from epoch to epoch
            random.seed(42)
            
            for _ in range(steps_per_epoch):
                # Renaming for more readability
                bs = batch_size
                isq = input_sequence_length
                osq = output_sequence_length
                
                # Generating pseudo random int that will be the starting index
                start = random.randint(0, features_array.shape[0] - ((bs*isq)+isq))
                #start = features_array.shape[0] - ((bs*isq)+isq) # for tests
                # Generating outputs
                encoder_input = features_array[start:(start+(bs*isq)), :].reshape(bs, isq, features_array.shape[1])
                decoder_output = label_array[start+isq:(start+isq+(bs*isq)), :].reshape(bs, isq, label_array.shape[1])[:,:osq,:]
                decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], label_array.shape[1])) 
                
                yield ([encoder_input, decoder_input], decoder_output)
```

### Function to prepare Validation inputs and Inference inputs of the RNN


```python
# This function will rearrange input data before using it in the model validation & prediction
# From 2D data (batch_size, data_dimension) to 3D data (batch_size, input_sequence_length, data dimension)
def prepare_3d_feature_data(previous_array, current_array, input_sequence_length, output_sequence_length):
    """
    - previous_array: 2D validation feature data (test use case) or 2D training data (validation use case) 
    - current_array: 2D test feature data (test use case) or 2D validation feature data (validation use case)  
    ==================================================================================
    Inference works like this (ex: input_sequence_length = 4 and output_sequence_length = 2) :
    (in1, in2, in3, in4) -> (pred5, pred6)
    (in5, in6, in7, in8) -> (pred9, pred10)
    pred7, pred8 is missing
    Given (in1, in2, in3, in4), (in5, in6, in7, in8) we need to generate the following data:
    (in1, in2, in3, in4), (in3, in4, in5, in6), (in5, in6, in7, in8), (in7, in8, in9, in10)
             |                     |                     |                     | 
       (pred5, pred6)       (pred7, pred8)         (pred9, pred10)      (pred11, pred12)
    That will give us continuous predictions that we need the model to generate
    Important note: The previous_array is used here as we will ned to take the last 4 elements and prepend them
    to the test data. This way we will be able to apply the model on these 4 elements and it will give the first 2 
    predictions.
    The algorithm used here will generate additional elements
    (they will be removed in the predict function just after the model predictions).
    The number of additional elements depends on the input_sequence_length, output_sequence_length combination
    """
    #input data length must be a multiple of input_sequence_length
    if len(current_array) % input_sequence_length != 0:
        raise Exception('input_sequence_length must be a multiple of current_array')
    elif len(previous_array) < input_sequence_length:
        raise Exception('previous_array is too small, last {} items needed - please check if mistake'.format(input_sequence_length))
        
    else:
        # Get last "input_sequence_length" elements of training data
        last_elts = previous_array[-input_sequence_length:]
        # Prepend the last "input_sequence_length" elements
        prepended_current_array = np.insert(current_array, 0, last_elts, axis=0)
        
        # Special case when input_sequence_length = output_sequence_length (no offset), we can return the reshaped input_data
        if input_sequence_length == output_sequence_length:
            # Return the test data prepended with the last "input_sequence_length" elements
            # Reshape it in order to send the right format to the Model (batch_size, input_sequence_length, 126)
            return np.reshape(prepended_current_array, (-1, input_sequence_length, current_array.shape[1]))
        
        else:
            offset = 0
            new_current_array = []
            while (offset + input_sequence_length) <= len(prepended_current_array):
                new_current_array.append(prepended_current_array[offset:offset+input_sequence_length])
                offset = offset + output_sequence_length
            # Reshape data in order to send the right format to the Model (batch_size, input_sequence_length, 126)
            return np.reshape(new_current_array, (-1, input_sequence_length, current_array.shape[1]))
        
```

### Function to prepare Validation labels


```python
# This function will rearrange label data for the model validation
# From 2D data (batch_size, data_dimension) to 3D data (batch_size, output_sequence_length, data dimension)
def prepare_3d_label_data( train_feature_array, 
                           validation_feature_array, 
                           validation_label_array, 
                           input_sequence_length, 
                           output_sequence_length):
    """
    - train_feature_array: 2D training feature data (to be passed to prepare_feature_data function)
    - validation_feature_array: 2D validation feature data
    - validation_label_array: 2D validation label data
    ==================================================================================
    Training works like this (ex: input_sequence_length = 4 and output_sequence_length = 2) :
    (in1, in2, in3, in4) -> (pred5, pred6)
    (in5, in6, in7, in8) -> (pred9, pred10)
    pred7, pred8 is missing
    Given (in1, in2, in3, in4), (in5, in6, in7, in8) we need to generate the following data:
    (in1, in2, in3, in4), (in3, in4, in5, in6), (in5, in6, in7, in8), (in7, in8, in9, in10)
             |                     |                     |                     | 
       (pred5, pred6)       (pred7, pred8)         (pred9, pred10)      (pred11, pred12)
    That will give us continuous labels that the model needs
    Validation input will look like: (pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12)
    This function will have to reshape it and make sure it has the right shape:
    (pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12)
                                |
    ((pred5, pred6), (pred7, pred8), (pred9, pred10), (pred11, pred12))
    The challenge here is that the expected model results might have a different size from the validation_label data
    (model result array of shape (batch_size=50, output_sequence_length=2, 1) -> 100 elements, and we have in input
    validation label data of shape (96, 1) -> 96 elements).
    We will then need to generate a model_validation_label_array that might have additional elements at 0 
    (if expected model results size > validation_label size)
    """
    # Prepare feature data
    feature_data = prepare_3d_feature_data(train_feature_array, 
                                           validation_feature_array, 
                                           input_sequence_length, 
                                           output_sequence_length)
    
    # Get batch size generated from features
    batch_size = feature_data.shape[0]
    
    # Calculate difference bw target size (nb of elements in validation) 
    # and expected size from model (batch_size*output_sequence_length)
    diff = validation_label_array.shape[0] - (batch_size*output_sequence_length)
    
    # Model generates more data than validation_label_array
    if diff < 0:
        # Append zeros
        extra_zeros = np.zeros((abs(diff), validation_label_array.shape[1]))
        validation_label_array_appended = np.append(validation_label_array, extra_zeros)
        
        # Reshape and return model_validation_array
        return np.reshape(validation_label_array_appended, (batch_size, 
                                                            output_sequence_length, 
                                                            validation_label_array.shape[1]))
        
    
    # Model generates exactly the same amount of data than validation_label_array (do nothing)
    elif diff == 0:
        # Reshape and return model_validation_array
        return np.reshape(validation_label_array, ( batch_size, 
                                                    output_sequence_length, 
                                                    validation_label_array.shape[1]))
    
    # Problem
    else:
        raise Exception('Model generates less data than validation data size')
        
```

### Fonction for Data update & preparation (train, val, test)


```python
# Utils function
def get_right_size(dataset_size, multiple, seq_len):
    # size greater or equal than product of multiple and seq_len
    if dataset_size - (multiple*seq_len) >= 0:
        ret_size = multiple*seq_len
    # size smaller than product of multiple and seq_len   
    else:
        ret_size = (multiple-1)*seq_len
        
    return ret_size


# DUE TO THE REQUIREMENTS OF THE SEQ2SEQ ALGORITHM:
# - train dataset size should be a multiple of the RNN Seq2Seq input_sequence_length
# - validation dataset size should be a multiple of the RNN Seq2Seq input_sequence_length
# - test dataset size should be a multiple of the RNN Seq2Seq input_sequence_length
# WE NEED A FUNCTION TO GIVE USE THE BEST SPLIT GIVEN:
# - input_sequence_length
# - train dataset 
# - validation datset
# - test datase
# DUE TO THE NATURE OF THE DATA (TIME SERIES), THE DATASETS MUST BE CONTINUOUS
def prepare_2d_splits(train_dataset, test_dataset, validation_split, input_sequence_length):
    """
    From the original datasets (train & test) and the validation split:
    - train_dataset: training array (feature or label)
    - test_dataset: test dataset (feature or label)
    - vaidation_split: fraction of training data to be used as validation (bw 0 and 1)
    - input_sequence_length: input_sequence_length for the Seq2Seq model
    
    Returns:
    - train_dataset: updated according to the validation split and the model requirements
    - validation_dataset: updated according to the validation split and the model requirements
    - test_dataset: updated according to the model requirements
    """
    # Initialize return values
    split_idx = int(train_dataset.shape[0]*(1-validation_split))
    ret_train_dataset = train_dataset[:split_idx, :]
    ret_val_dataset = train_dataset[split_idx:, :]
    ret_test_dataset = test_dataset
    
    # Sizes
    train_dataset_size = ret_train_dataset.shape[0]
    val_dataset_size = ret_val_dataset.shape[0]
    test_dataset_size = ret_test_dataset.shape[0]
    
    # Concatenate train & test
    dataset = np.concatenate((train_dataset, test_dataset), axis=0)
    
    # Calculate multiples
    multiple_train = int(train_dataset_size/input_sequence_length)
    multiple_val = int(val_dataset_size/input_sequence_length)
    multiple_test = int(test_dataset_size/input_sequence_length)
    
    # train data
    train_dataset_size = get_right_size(train_dataset_size, multiple_train, input_sequence_length)   
    ret_train_dataset = dataset[:train_dataset_size, :]
    
    # validation dat
    val_dataset_size = get_right_size(val_dataset_size, multiple_val, input_sequence_length)
    ret_val_dataset = dataset[train_dataset_size:(train_dataset_size+val_dataset_size), :]
        
    # test data
    test_dataset_size = get_right_size(test_dataset_size, multiple_test, input_sequence_length)
    ret_test_dataset = dataset[(train_dataset_size+val_dataset_size):(train_dataset_size+val_dataset_size+test_dataset_size), :]
    
    return (ret_train_dataset, ret_val_dataset, ret_test_dataset)

```

## Model


```python
import keras
```

    Using TensorFlow backend.
    

### Hyperparameters


```python
keras.backend.clear_session()

num_layers = 2
hidden_neuros = 32

layers = [hidden_neuros]*num_layers # [32 ..] Number of hidden neuros in each layer of the encoder and decoder
learning_rate = 0.001
decay = 0 # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate)#keras.optimizers.Adam(lr=learning_rate, decay=decay)
rnn_dropout = 0.9 # % of neuros to turn off while training RNN Cells
rnn_recurrent_dropout = 0.9 # % of neuros to turn off while training RNN Cells
dense_dropout = 0.9 # % of neuros to turn off while training Dense network

num_input_features = 126 # The dimensionality of the input at each time step.
num_output_features = 1 # The dimensionality of the output at each time step.
# There is no reason for the input sequence to be of same dimension as the ouput sequence:
# Input has 126 features (erros, ticks, mean ...) while output label has 1 feature (machine OK or NOK)

loss = "binary_crossentropy"#"binary_crossentropy" # Other loss functions are possible, see Keras documentation.
activation_dense = "sigmoid" #Activation function for the dense layer

# Regularisation isn't really needed for this application
lambda_regulariser = 0.000001 # Will not be used if regulariser is None 0.000001
regulariser = None#keras.regularizers.l2(lambda_regulariser)

#Weight initializers
rnn_kernel_initializer='glorot_uniform'  #default: 'glorot_uniform' 
rnn_recurrent_initializer= 'orthogonal'  #default: 'orthogonal' 
dense_kernel_initializer='random_uniform'#default: 'random_uniform' 

batch_size = 240 #240 80
steps_per_epoch = 200 #200 # batch_size * steps_per_epoch = total number of training examples
epochs = 20

input_sequence_length = 1 # Length of the sequence used by the encoder
output_sequence_length = 1 # Length of the sequence predicted by the decoder

validation_split = 0.1
```

Our model will take the 126 inputs (errors , error ticks, error tick means ...) and will output the label (1 machine is KO, 0 machine is OK).

We will then use the Seq2Seq to do classification here (machine OK or KO). As we are dealing with a binary classification, we chose to use the binary_crossentropy loss function.

We could have created a custom ROC-AUC metric but it was not working properly and the training could not be launched with this custom metric.

The AUC-ROC will be calculated on the test dataset after the mode training.

### Final train, validation, test split


```python
# Prepare 2d splits that complies with model requirements (right sizes)
X_train_2d, X_val_2d, X_test_2d = prepare_2d_splits(X_train_np, X_test_np, validation_split, input_sequence_length)
y_train_2d, y_val_2d, y_test_2d = prepare_2d_splits(y_train_np, y_test_np, validation_split, input_sequence_length) 


# Prepare validation_data needed during fit
X_val_3d = prepare_3d_feature_data(X_train_2d, X_val_2d, input_sequence_length, output_sequence_length)
# print(X_val_3d.shape)
y_val_3d = prepare_3d_label_data( X_train_2d, 
                                  X_val_2d, 
                                  y_val_2d, 
                                  input_sequence_length, 
                                  output_sequence_length)

# Create a decoder input with same dimensions as decoder outputs (y_val_3d) and all values at 0
X_decoder_zeros_3d = (np.zeros((y_val_3d.shape[0], y_val_3d.shape[1], y_val_3d.shape[2]))).astype(np.float32)
```

### Encoder


```python
# Define an input sequence.
encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.LSTMCell(hidden_neurons,
                                               kernel_initializer=rnn_kernel_initializer, 
                                               recurrent_initializer=rnn_recurrent_initializer,
                                               kernel_regularizer=regulariser,
                                               recurrent_regularizer=regulariser,
                                               bias_regularizer=regulariser,
                                               dropout=rnn_dropout,
                                               recurrent_dropout=rnn_recurrent_dropout))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = encoder_outputs_and_states[1:]
```

### Decoder


```python
# The decoder input will be set to zero (see batch_generator function above).
decoder_inputs = keras.layers.Input(shape=(None, num_output_features))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.LSTMCell(hidden_neurons,
                                               kernel_initializer=rnn_kernel_initializer, 
                                               recurrent_initializer=rnn_recurrent_initializer,
                                               kernel_regularizer=regulariser,
                                               recurrent_regularizer=regulariser,
                                               bias_regularizer=regulariser,
                                               dropout=rnn_dropout,
                                               recurrent_dropout=rnn_recurrent_dropout))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
decoder_dense = keras.layers.Dense(num_output_features,
                                   activation=activation_dense,
                                   kernel_initializer=dense_kernel_initializer,
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)


decoder_outputs = decoder_dense(decoder_outputs)

final_decoder_outputs = keras.layers.Dropout(dense_dropout)(decoder_outputs)
```

### Create and compile Model


```python
model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)
```


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, 126)    0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            (None, None, 1)      0                                            
    __________________________________________________________________________________________________
    rnn_1 (RNN)                     [(None, 32), (None,  28672       input_1[0][0]                    
    __________________________________________________________________________________________________
    rnn_2 (RNN)                     [(None, None, 32), ( 12672       input_2[0][0]                    
                                                                     rnn_1[0][1]                      
                                                                     rnn_1[0][2]                      
                                                                     rnn_1[0][3]                      
                                                                     rnn_1[0][4]                      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, None, 1)      33          rnn_2[0][0]                      
    ==================================================================================================
    Total params: 41,377
    Trainable params: 41,377
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

### Fit Model to Data


```python
train_data_generator = batch_generator( X_train_2d, 
                                        y_train_2d, 
                                        steps_per_epoch = steps_per_epoch,
                                        batch_size = batch_size, 
                                        input_sequence_length = input_sequence_length, 
                                        output_sequence_length = output_sequence_length)

history = model.fit_generator(train_data_generator, 
                              steps_per_epoch=steps_per_epoch, 
                              epochs=epochs, validation_data=([X_val_3d, X_decoder_zeros_3d], y_val_3d))
```

    Epoch 1/20
    200/200 [==============================] - 6s 28ms/step - loss: 0.3990 - val_loss: 0.5992
    Epoch 2/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2167 - val_loss: 0.5529
    Epoch 3/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2099 - val_loss: 0.5370
    Epoch 4/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2079 - val_loss: 0.5273
    Epoch 5/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2046 - val_loss: 0.5304
    Epoch 6/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2047 - val_loss: 0.5259
    Epoch 7/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2030 - val_loss: 0.5252
    Epoch 8/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2015 - val_loss: 0.5256
    Epoch 9/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2017 - val_loss: 0.5227
    Epoch 10/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2009 - val_loss: 0.5244
    Epoch 11/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1998 - val_loss: 0.5293
    Epoch 12/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.2008 - val_loss: 0.5275
    Epoch 13/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1998 - val_loss: 0.5262
    Epoch 14/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1990 - val_loss: 0.5312
    Epoch 15/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1982 - val_loss: 0.5274
    Epoch 16/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1984 - val_loss: 0.5315
    Epoch 17/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1973 - val_loss: 0.5377
    Epoch 18/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1974 - val_loss: 0.5398
    Epoch 19/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1967 - val_loss: 0.5386
    Epoch 20/20
    200/200 [==============================] - 2s 9ms/step - loss: 0.1969 - val_loss: 0.5390
    

### Train and Validation loss


```python
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
```


![png](capstone_project_files/capstone_project_99_0.png)


We can clearly see that the training loss decreases steadily and that the validation loss decreases as well but much less.
At 50 epochs the training and validation losses seem to stabilize.

We see no signs of underfitting but the validation loss increases slightly after 7 epochs which can be a sign of overfitting. The increase is very small and the training was stopped before the underfitting is clearly visible.


**Final training loss: 0.1979**

**Final validation loss: 0.5423**

### Save model

### Predict Function for testing


```python
# Creating a predict function that will use the model to predict in order to test
def predict(previous_features_array, test_features_array, model, input_sequence_length, output_sequence_length):
    """
    Predicts time series according to the given model. 
    Rearranges the test_features_array to predict for a continuous window corresponding to the test labels array.
    Explanation: 
    Our model takes "input_sequence_length" elements (ex: 4) and predicts the next "input_sequence_length" elements (ex: 2),
    it then takes the next 4 elements and predicts the next 2
    but here the next 2 outputs are not the just after the 2 predictions. There is an offset of 2 elements between the predictions:
    (in1, in2, in3, in4) -> (pred5, pred6)
    (in5, in6, in7, in8) -> (pred9, pred10)
    out7 and out8 are not available if we apply the model to the data directly. We need to prepare the data in order to
    also apply (in3, in4, in5, in6) -> (pred7, pred8)
    
    Other important note: 
    We can not just feed the model with the above data (in1, in2, in3, in4) and then (in5, in6, in7, in8) ....
    We need to feed the 4 elements prior to the needed predictions, the 4 elements before (in1, in2, in3, in4) in
    order to have (pred1, pred2) ...
    That is why the trainig data is an input to this function, we need the last 4 elements of the training data.
    - previous_features_array: whole taining data (train + val) of shape (number of elements in the  whole training data, dimension of element)
    - test_features_array: testing data of shape (number of elements in the testing data, dimension of element) 
    - model_path: String pointing to the path of best model
    - input_sequence_length: used to prepare the data
    - output_sequence_length: use to prepare the data
    """
    # Preparing 3D feature data
    X_train_ready = prepare_3d_feature_data(previous_features_array, 
                                            test_features_array, 
                                            input_sequence_length, 
                                            output_sequence_length)
    decoder_inputs = np.zeros((X_train_ready.shape[0], output_sequence_length, 1))
    
    # Applying model
    #saved_model = keras.models.load_model(model_path)
    y_pred = model.predict([X_train_ready, decoder_inputs])
    
    # Reshaping from 3D to 2D (Model generates 3D predictions)
    y_pred_reshaped = np.reshape(y_pred, (-1, num_output_features)) # element dimension is num_output_features
                                     
    # Our data preparation implies an additional "output_sequence_length" predictions 
    # We then need to get only the number of elements corresponding to the test label dataset which is the same number
    # as the elements in test feature dataset (basically the number of days).
    return y_pred_reshaped[:test_features_array.shape[0], :]
    
```

### Evaluate model


```python
# Run prediction on the testing dataset:
#   The previous dataset is X_train. But it was split in X_train_2d and X_val_2d. The last training data is 
#   then the validation data. We will provide the validation data (X_val_2d_prepared) to the predict function 
#   as it needs the data before the testing data. 
y_test_pred = predict(X_val_2d, X_test_2d, model, input_sequence_length, output_sequence_length)

print(y_test_pred.shape)
print(y_test_pred)
```

    (137, 1)
    [[0.03673244]
     [0.02324419]
     [0.02932579]
     [0.03300719]
     [0.01705262]
     [0.03755571]
     [0.03281975]
     [0.03446746]
     [0.03441212]
     [0.03292985]
     [0.03477302]
     [0.03625613]
     [0.03552966]
     [0.01404711]
     [0.01561557]
     [0.02343476]
     [0.03678703]
     [0.0362521 ]
     [0.037721  ]
     [0.04214078]
     [0.03674681]
     [0.03663356]
     [0.0144521 ]
     [0.03719785]
     [0.03540173]
     [0.03918133]
     [0.04489202]
     [0.03572749]
     [0.03275581]
     [0.02033077]
     [0.04176112]
     [0.01744358]
     [0.03289427]
     [0.0273035 ]
     [0.03248088]
     [0.03487169]
     [0.01734845]
     [0.01961147]
     [0.01733006]
     [0.03963321]
     [0.04074971]
     [0.03183097]
     [0.0315209 ]
     [0.03698122]
     [0.01563516]
     [0.02169936]
     [0.01705518]
     [0.03410702]
     [0.03729011]
     [0.01662798]
     [0.01684005]
     [0.03676129]
     [0.01559008]
     [0.03407116]
     [0.01588629]
     [0.03429263]
     [0.03251246]
     [0.01318131]
     [0.0338605 ]
     [0.02202608]
     [0.0162917 ]
     [0.01365875]
     [0.01440162]
     [0.02127027]
     [0.0194149 ]
     [0.01494688]
     [0.01795783]
     [0.03576927]
     [0.03435282]
     [0.01559492]
     [0.02261013]
     [0.0171525 ]
     [0.01793041]
     [0.03259152]
     [0.03465567]
     [0.01589409]
     [0.02775429]
     [0.03521733]
     [0.01949606]
     [0.03438527]
     [0.04169312]
     [0.03660284]
     [0.03299175]
     [0.03976256]
     [0.02066733]
     [0.0162673 ]
     [0.03570217]
     [0.03938777]
     [0.01593043]
     [0.03310807]
     [0.0156597 ]
     [0.03342811]
     [0.0336443 ]
     [0.02149312]
     [0.02061233]
     [0.01902048]
     [0.02206716]
     [0.04063492]
     [0.03647446]
     [0.01922309]
     [0.03443138]
     [0.04185406]
     [0.03310755]
     [0.04614222]
     [0.03562326]
     [0.04436945]
     [0.03353851]
     [0.04052418]
     [0.01999619]
     [0.04753719]
     [0.05316829]
     [0.01842638]
     [0.01858259]
     [0.0186631 ]
     [0.0193484 ]
     [0.01887677]
     [0.01940108]
     [0.0204171 ]
     [0.01853082]
     [0.02145838]
     [0.01919962]
     [0.03507942]
     [0.03650622]
     [0.04890378]
     [0.03578076]
     [0.03567678]
     [0.0353627 ]
     [0.04852926]
     [0.03702879]
     [0.03798857]
     [0.03673157]
     [0.03799254]
     [0.01554524]
     [0.02295701]
     [0.0456454 ]
     [0.0438447 ]
     [0.03797616]]
    


```python
# Plot predicted and actual
plt.figure(figsize=(15, 7))
plt.plot(y_test_pred, label="predicted label")
plt.plot(y_test_2d, label="actual label") 

# Titles, Legends and formatting
plt.title("actual vs predicted label")
plt.grid(True)
plt.legend(ncol=2, loc='upper right')
plt.show()
```


![png](capstone_project_files/capstone_project_107_0.png)


## AUC for RNN Algorithm


```python
# Calculate auc
roc_auc_score(y_test_2d, y_test_pred) 
```




    0.6129032258064516



# 7.Comparison of Baseline Algorithm (VAR) and Selected Algorithm (Seq2Seq)

## Summary of the Baseline Algorithm (VAR)

The implemented VAR algorithm has many **advantages**:
- **Model simplicity**: No need for feature preparations.
- **Prediction of label and features**: Pure time-series prediction.
- **Well known method**: Old method with very well known formulas and improvements (statsmodel library).
- **Deterministic**: Running the algorithm many times with the same data gives the same result every time.

But our VAR implementation also have some **drawbacks**:
- **High probability of divergence**: Predicted features are used to predict label instead of real features (this needs to be improved with feature preparation).
- **Poor results for long predictions**: When deciding to predict many days at once, the results were very poor. So we decided to use this model to predict one day at most. The cost was running the algorithm for every day predicted (Performance cost).
- **Performance**: The implemented VAR needs to be running for every day predicted which is very costly. The model is not a machine learning algorithm so it processes all data to predict one day (No concepts of trainig and inference). It is not scalable. As an example, here it takes 5 to 10 minutes to predict 137 days.

### Final results:

#### **ROC-AUC score: 0.597**

<img src="VAR_result.png">

**Qualitative analysis**: 

The quality of the result is quite good, we can clearly see that the predicted peaks are higher near the actual failures. The only problem is that the predicted peaks seem to have a delay and are not exactly synchronized with the actual failures.

## Summary of the Selected Algorithm (RNN Seq2Seq)

The implemented Seq2Seq algorithm has many **advantages**:
- **Performance**: Can predict many days given only the previous X days (depending on the input sequence). Scalable. Can predict effectively many days in advance. Training and Inference are separated (Inference is really fast - less than one minute for our implemented version). It takes less than 10 seconds to predict 137 days here.
- **High probability of convergence**: The model always uses the real features to predict labels.
- **Designed for long predictions**: The model can learn to predict many days in advance (output sequence of many days) with the same probability of the first day (model parameters will reflect the long prediction).
- **Recent method but well documented**: Many libraries implementing it (Keras, Tensorflow, Pytorch, MXNet ...) and other libraries facilitating hyperparameters search (SMAC3 - not implemented here due to lack of time).
- **Prediction flexibility**: Can be used to predict time series only or labels as a classification problem (that is what we choose here).

We can clearly see here that the Seq2Seq model covers all the drawbacks of the VAR algorithm and it also shares some advantages with the VAR algotihm. But we will see that it also has drawbacks that are covered by the VAR algorithm (See next).

But Seq2Seq **drawbacks** are also as follows:
- **Model complexity**: Many hyperparameters to tweak and heavy data preparation needed.
- **Non Deterministic**: Running the algorithm many times with the same data gives slightly different results every time.

### Final results

#### **ROC-AUC score: 0.613**

<img src="Seq2Seq_result.png">

## Conclusion

In theory, the Seq2Seq model seems to be the best of both models as it has many advantages of the VAR but it comes with a cost which is complexity and non determinism.

And this is confirmed when we compare the results quantitaively: we see a small ROC-AUC improvement (2.5% improvement) for the Seq2Seq model. But the small improvement came with a lot of effort for data preparation and hyperparameter tunning.

Even though the qualitative comparison is in favor of the VAR algorithm, the qualitative comparison is very explicitly going in favor of the VAR algorithm. Compared to the Seq2Seq model, the VAR models clearly shows higher peaks near the actual failure days. Instead, the Seq2Seq model tried to have the best possible binary cross entropy loss even it that means havng an almost flat prediction.

The Seq2Seq model needs more data and more tunning to be abe to give results that are qualitatively correct. I tried to use another loss technique: Mean Squared Error (MSE) in order to have better qualitative results but the ROC-AUC and the quality of the results were just slightly higher. I then choose to keep the Binary Cross entropy loss as it is a categorical loss. The MSE is a distance indicator that should be used for pure time series prediction (regression) and not for classification.

The poor results of the Seq2Seq model are most likely due to the small dataset size and the high dimensionality. I have no doubt that more data and more feature engineering (reducing the number of features by combining them) will give substantially better results for the Seq2Seq model.

As an indication, the leaderboard of the Kaggle competition ([leaderboard](https://www.kaggle.com/c/predictive-maintenance1/leaderboard)) shows that a result of 0.613 would be at the third position. 

My ROC-AUC was not calculated on the same data so we can not compare but that shows an indication of the difficulty of this problem and the corresponding dataset.
