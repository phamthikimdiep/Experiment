# ----------------------------//-------------------------FUNCTIONS
import math


from Functions import *
warnings.filterwarnings("ignore")

# number of folds
n_splits = 5

#### DATASETS ####
# 3 -Credit Card: giao dich the tin dung ############
#     - Trung binh cac khoan giao dich la 88 USD    #
#     - Khong co gia tri NULL                       #
#     - Hau het cac giao dich la Non-Fraud (99.83%) #
#      con lai la Fraud (0.17%)                     #
#####################################################

print("\n ========== Dataset Info ========== \n")
file_creditcard = 'data/creditcard.csv'
creditcard_data = pd.read_csv(file_creditcard, header=0, engine='python')
print(creditcard_data.head())
print(creditcard_data.columns)
print(creditcard_data.describe())

# Good No Null Values!
creditcard_data.isnull().sum().max()

# remove special characters
remove(file_creditcard, '\/:*?"<>|')

#set index for columns
creditcard_data = pd.DataFrame(creditcard_data)

print('\n---------------The shape of data')
detail_data(creditcard_data, '---------------Info of Credit Fraud detector')
print(creditcard_data.describe())
print(creditcard_data.head())
print('Dataset - Original')
print(creditcard_data.info())

print('---------------checking the percentage of missing data contains in all the columns')

missing_percentage = creditcard_data.isnull().sum()/creditcard_data.shape[0]
print('\n ---------------- Missing percentage: \n{}'.format(missing_percentage))

total = creditcard_data.isnull().sum().sort_values(ascending=False)
percent = (creditcard_data.isnull().sum()/creditcard_data.isnull().count()*100).sort_values(ascending=False)
pd.concat([total,percent],axis=1, keys = ['Total','Percent']).transpose()

# Imputing Missing Values with 0
print('\n--------------- Imputing Missing Values with 0')
creditcard_data.fillna(0,inplace=True)


##########################
# CHECK THE DISTRIBUTION #
##########################

print('\n##########################\n# CHECK THE DISTRIBUTION #\n##########################')


creditcard_target = ['Class']

# colunms of data
creditcard_cols = [i for i in creditcard_data.columns if i not in creditcard_target]

# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(creditcard_data['Class'].value_counts()[0]/len(creditcard_data) * 100,2), '% of the dataset')
print('Frauds', round(creditcard_data['Class'].value_counts()[1]/len(creditcard_data) * 100,2), '% of the dataset')
'''
#Distribution of Customers on Revenue - Shopper
plot_all_distribution_bar(creditcard_data.Class.value_counts(),
                          'Distribution of Class - CreditCard',
                          'No Frauds / Frauds', 'Counts')
'''
#Transaction Amount and Time
print('\n{}\n#   Transaction of Amount and Time   #\n{}\n'.format('#'*35,'#'*35))
print(creditcard_data[['Time','Amount']].describe())

amount_val = creditcard_data['Amount'].values
print('\n*** Amount\nMax value: {} - Min value: {}'.format(max(amount_val),min(amount_val)))
time_val = creditcard_data['Time'].values
print('\n*** Time\nMax value: {} - Min value: {}'.format(max(time_val),min(time_val)))
fig, ax = plt.subplots(1, 2, figsize=(18,6))

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Amount Transaction', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Time Transaction', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
#plt.savefig('results_fig/Distribution of Amount and Time Transaction.png')

# Distribution of Amount (individual) - Fraud and Not-Fraud transactions
fig, axs = plt.subplots(ncols=2, figsize =(18,6))
sns.distplot(creditcard_data[creditcard_data['Class'] == 1]['Amount'],bins=100,ax = axs[0])
axs[0].set_title('Distribution of Fraud Transactions')

sns.distplot(creditcard_data[creditcard_data['Class'] == 0]['Amount'],bins=100, ax = axs[1])
axs[1].set_title('Distribution of Not-Fraud Transactions')
#plt.savefig('results_fig/Distribution of Amount Transaction - Individual.png')

# Distrubution of Amount with respect to (w.r.t) Class || Amount => Class
plt.figure(figsize=(8,6))
sns.boxplot(x='Class', y = 'Amount', data = creditcard_data)
plt.title('Distribution of Amount with respect to Class')
#plt.savefig('results_fig/Distribution of Amount with respect to Class.png')

# Distribution of Time (individual) - Fraud and Not-Fraud transactions
fig, axs = plt.subplots(ncols=2, figsize=(18,6))
sns.distplot(creditcard_data[creditcard_data['Class']==1]['Time'], bins=100, color='red', ax=axs[0])
axs[0].set_title('Distribution of Fraud Transactions')

sns.distplot(creditcard_data[creditcard_data['Class'] == 0]['Time'], bins=100, color = 'green', ax=axs[1])
axs[1].set_title('Distribution of Not-Fraud Transactions')
#plt.savefig('results_fig/Distribution of Time Transaction - Individual.png')

# Distrubution of Time with respect to (w.r.t) Class || Time => Class
plt.figure(figsize=(12,8))
ax = sns.boxplot(x='Class', y = 'Time', data=creditcard_data)
plt.title('Distribution of Time with respect to Class')
#plt.savefig('results_fig/Distribution of Time with respect to Class.png')

# Distribution of Class with respect to (w.r.t) Amount_Time || Class => Amount_Time
fig, axs = plt.subplots(ncols=2, figsize = (16,6))
sns.scatterplot(x='Time', y = 'Amount', data = creditcard_data[creditcard_data['Class'] == 1], ax = axs[0])
axs[0].set_title('Distribution of Fraud Transactions')

sns.scatterplot(x='Time', y = 'Amount', data = creditcard_data[creditcard_data['Class'] == 0], ax = axs[1])
axs[1].set_title('Distribution of Not-Fraud Transactions')
#plt.savefig('results_fig/Distribution of Class with respect to Amount_Time')

#plt.show()

#Finding unique values for each column to understand
# which column is categorical and which one is Continuous

# Figuring unique values for each column
print('\n--------------- Figuring unique values for each columns: \n')
print(creditcard_data[['Time', 'Amount', 'Class']].nunique())

# Check the data again after cleaning
print('\n--------------- Check the data again after cleaning\n')

print('*** Shape of data: {}'.format(creditcard_data.shape))
print('*** Counts of Class values - normalize = True: \n{}'.format(creditcard_data['Class'].value_counts(normalize=True)))

##########################
#   FEATURE EXTRACTION   #
##########################
print('\n{}\n#   FEATURE EXTRACTION   #\n{}\n'.format('#'*26,'#'*26))

#converting time from second to hour
print('-'* 10 + ' converting time from second to hour.')
creditcard_data['Time'] = creditcard_data['Time'].apply(lambda sec : (sec/3600))

#calculating hour of the day
print('-'* 10 + ' calculating hour of the day.')
creditcard_data['Hour'] = creditcard_data['Time']%24 #2 days of data
creditcard_data['Hour'] = creditcard_data['Hour'].apply(lambda x : math.floor(x))

#calculating 1st and 2nd day
print('-'*10 + ' calculating 1st and 2nd day.')
creditcard_data['Day'] = creditcard_data['Time']/24 #2 days of data
creditcard_data['Day'] = creditcard_data['Day'].apply(lambda x : 1 if(x==0) else math.ceil(x))

#data after converting and calculating time
print('\n' + '-'*10 + ' data after converting and calculating time')
print(creditcard_data[['Time','Hour','Day','Amount','Class']])

#Fraud and Genuine transaction Day wise
print('\n' + '-'*10 + '\nFraud and Genuine transactions Day wise\n' + '-'*10)

#calculating fraud transaction daywise
dayFraudTran = creditcard_data[(creditcard_data['Class'] == 1)]['Day'].value_counts()
print('*** Day Fraud Transactions:\n{}\n'.format(dayFraudTran))

#calculating not-fraud transaction daywise
dayNotFraudTran = creditcard_data[(creditcard_data['Class'] == 0)]['Day'].value_counts()
print('*** Day Not-Fraud Transactions:\n{}\n'.format(dayNotFraudTran))

#calculating total transaction daywise
dayTran = creditcard_data['Day'].value_counts()
print('*** Total Transaction Daywise:\n{}\n'.format(dayTran))

#==> percentage of fraud transactions day wise:
print('\n Percentage of Fraud Transactions Day Wise: \n{}'.format(round((dayFraudTran/dayTran)*100,2)))

#==> view on Graph
fig, axs = plt.subplots(ncols=3, figsize = (18,6))

#total transactions
sns.countplot(creditcard_data['Day'], ax = axs[0])
axs[0].set_title("Distribution of Total Transactions")

#Fraud transactions
sns.countplot(creditcard_data[(creditcard_data['Class'] == 1)]['Day'], ax = axs[1])
axs[1].set_title("Distribution of Fraud Transactions")

#Not-Fraud transactions
sns.countplot(creditcard_data[(creditcard_data['Class'] == 0)]['Day'],ax = axs[2])
axs[2].set_title("Distribution of Not-Fraud Transactions")

#plt.savefig('results_fig/View_Of_Transactions_Day_Wise.png')
#plt.show()

#Comparison between Transaction Frequencies vs Time for Fraud and Not-Fraud transactions

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,8))

#present - Not-Fraud Transactions
sns.distplot(creditcard_data[creditcard_data['Class'] == 0]['Time'].values, color='green', ax = axs[0])
axs[0].set_title("Not-Fraud Transactions")

#present - Fraud Transactions
sns.distplot(creditcard_data[creditcard_data['Class'] == 1]['Time'].values, color='red', ax = axs[1])
axs[1].set_title("Fraud Transactions")

fig.suptitle('Comparison between Transaction Frequencies vs Time for Fraud and Not-Fraud Transactions')
#plt.savefig('results_fig/Comparison_Freq_vs_Time_Trans.png')
#plt.show()

#overview - time, hour, day, amount, class Group by Hour - Count by Class
plt.figure(figsize=(8,6))
creditcard_data[['Time', 'Hour', 'Day', 'Amount', 'Class']].groupby('Hour').count()['Class'].plot()
#plt.savefig('results_fig/Overview_Time_Amount_groupbyHour.png')
#plt.show()

#reset index
creditcard_data.reset_index(inplace=True, drop=True)

############################
#   SCALE AMOUNT FEATURE   #
############################
print('\n{}\n#   SCALE AMOUNT FEATURE   #\n{}\n'.format('#'*28,'#'*28))

#Scale amount by log
print('-'*10 + ' Scale amount by log.')

# Adding a small amount of 0.0001 to amount as log of zero is infinite.
creditcard_data['Amount_log'] = np.log(creditcard_data.Amount + 0.0001)

#Scale amount by Standardization
print('-'*10 + ' Scale amount by Standardization.')

scaler = StandardScaler()
creditcard_data['Amount_scaled'] = scaler.fit_transform(creditcard_data['Amount'].values.reshape(-1,1))

#Scale amount by Normalization
print('-'*10 + ' Scale amount by Normalization.')

mm = MinMaxScaler()
creditcard_data['Amount_minmax'] = mm.fit_transform(creditcard_data['Amount'].values.reshape(-1,1))

print(creditcard_data[['Time', 'Hour', 'Day', 'Amount', 'Amount_log', 'Amount_scaled', 'Amount_minmax','Class']])

print(creditcard_data.head(5))

print(creditcard_data.columns)


#Columns for visualization purpose only - not for build model will be dropped
# Hour, Day, Scaled amount columns
cols_to_drop = ['Hour', 'Day', 'Amount_minmax', 'Amount_scaled', 'Amount_log']
creditcard_data = creditcard_data.drop(cols_to_drop,axis=1)
print(creditcard_data.columns)


#########################################
#          DATA PREPROCESSING           #
#########################################

print('#' *39 + '\n#          DATA PREPROCESSING           #\n' + '#' * 39)

#target_column
target_col = ['Class']
print('\n - Target columns: ',target_col)

#num cols
cols_to_scale = ['Time','Amount']
#cols_to_scale = [i for i in creditcard_data.columns if i not in target_col]
print('Num cols: ',cols_to_scale)

#Binary columns with 2 values
bin_cols = creditcard_data.nunique()[creditcard_data.nunique() == 2].keys().tolist()
print('Bin_Cols: ',bin_cols)

#label encoding binary columns
le = LabelEncoder()
for i in bin_cols:
    creditcard_data[i] = le.fit_transform(creditcard_data[i])

#scaling numerical columns
std = StandardScaler()
scaled = std.fit_transform(creditcard_data[cols_to_scale])
scaled = pd.DataFrame(scaled, columns=cols_to_scale)

#dropping original values - merging scaled values
df_creditcard_data_og = creditcard_data.copy()
creditcard_data = creditcard_data.drop(columns = cols_to_scale, axis = 1)
creditcard_data = creditcard_data.merge(scaled, left_index=True, right_index=True, how = 'left')
#creditcard_data = creditcard_data.merge(scaled, left_index=True, right_index=True, how = 'inner')

print(creditcard_data.head())
print(creditcard_data.columns)

##############################
###    VARIABLE SUMMARY    ###
##############################

summary = (df_creditcard_data_og[[i for i in df_creditcard_data_og.columns]].describe().transpose().reset_index())
summary = summary.rename(columns = {'index':'feature'})
summary = np.around(summary,3)

val_lst = [summary['feature'],
           summary['count'],
           summary['mean'],
           summary['std'],
           summary['min'],
           summary['25%'],
           summary['50%'],
           summary['75%'],
           summary['max']]

trace = go.Table(header=dict(values = summary.columns.tolist(),
                             line = dict(color=['#506784']),
                             fill = dict(color = ['#119DFF'])),
                 cells=dict(values=val_lst,
                            line = dict(color=['#506784']),
                            fill = dict(color=['lightgrey','#F5F8FF'])),
                 columnwidth=[200,60,100,100,60,60,80,80,80])
layout = go.Layout(dict(title='Training variable summary'))
figure = go.Figure(data = [trace], layout=layout)
#py.plot(figure, filename='results_fig/credit_training_variable_summary.html')

##############################
###    CORRELATION MATRIX    ###
##############################

correlation = creditcard_data.corr()
plt.figure(figsize=(14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, linewidths = .1, cmap='Reds')
#plt.savefig('results_fig/credit_correlation.png')



#PCA
pca = PCA(n_components=2, random_state=42)

X = creditcard_data[[i for i in creditcard_data.columns if i not in target_col]]
Y = creditcard_data[target_col]

principal_components = pca.fit_transform(X)
pca_data = pd.DataFrame(principal_components, columns=['PC1','PC2'])
pca_data = pca_data.merge(Y, left_index=True, right_index=True, how='left')
pca_data['Class'] = pca_data['Class'].replace({1:'Fraud', 0:'Not Fraud'})
sns.relplot(x='PC1', y = 'PC2',
            hue = 'Class', alpha = .5,
            palette = 'muted', height=6,
            data = pca_data)
#plt.savefig('results_fig/credit_pca.png')
#plt.show()



########################################
#   SPLITTING DATA INTO TRAIN - TEST   #
########################################
print('\n{}\n#   SPLITTING DATA INTO TRAIN - TEST   #\n{}\n'.format('#'*40,'#'*40))


#drop time columns
time_cols = ['Time']
creditcard_data = creditcard_data.drop(time_cols,axis = 1)
print('Columns: \n{}'.format(creditcard_data.columns))
print(creditcard_data.head())


train, test = train_test_split(creditcard_data, test_size=.25, random_state=42)


#seperating dependent and independent variables

cols = [i for i in creditcard_data.columns if i not in target_col]

X_train = train[cols]
Y_train = train[target_col]
X_test = test[cols]
Y_test = test[target_col]

print('*** Shape of X_train: \n{}'.format(X_train.shape))
print('*** Shape of Y_train: \n{}'.format(Y_train.shape))
print('*** Shape of X_test: \n{}'.format(X_test.shape))
print('*** Shape of Y_test: \n{}'.format(Y_test.shape))

##############################
###     MODEL BUILDING     ###
##############################
print('\n##############################\n###     MODEL BUILDING     ###\n##############################')


X = creditcard_data[cols]
Y = creditcard_data[target_col]

#-----------------
'''
#variance Threshold for feature selection
var = VarianceThreshold(threshold=.5)
var.fit(X,Y)
X_var = var.transform(X)
print('X_var shape: ',X_var.shape)

mask = var.get_support()
print('Var: ',mask)
plt.matshow(mask.reshape(-1,1))
#plt.show()



#Select Percentile
select50 = SelectPercentile(percentile=50)
select50.fit(X,Y)
X_select50 = select50.transform(X)
print('\nShape of X 50%: {}'.format(X_select50.shape))

#look at features selected by select percentile using a boolean mask
mask50 = select50.get_support()
print('\nMask 50%\n {}'.format(mask50))
#selected_values = X.columns.values[mask]
print('-'*40+'\n')
#print('cols of dataset: ', creditcard_data.columns)
#print('selected cols: ',selected_values)

#cols = [i for i in selected_values]


plt.matshow(mask50.reshape(1,-1))
plt.title('Selected Features 50%')
plt.savefig('results_fig/SelectedFeatures50.png')

print('-'*40 +'\n')

feature50 = X.columns.values[mask50]
print('\nfeature50 cols: ',feature50)


print('\n' + '-'*20 + 'Choose - 50%' + '-'*20)
#choose 50%
X_select = X_select50
selected_values = X.columns.values[mask50]

print('\ncols of dataset: ', creditcard_data.columns)
print('\nselected cols: ',selected_values)

cols = [i for i in selected_values]
'''

#plt.show()

#select 10 best
skbest = SelectKBest(k=10)
#skbest = SelectPercentile(percentile=50)
skbest.fit(X, Y)
X_kbest = skbest.transform(X)
print(X_kbest.shape)

mask_kbest = skbest.get_support()
print('\n True values are selected features')
print('Kbest: ',mask_kbest)
plt.matshow(mask_kbest.reshape(1,-1))
plt.title('K Best Selected Features',fontsize=13)
#plt.savefig('results_fig/10bestSelectedFeatures.png')
#plt.show()

print('\n' + '-'*20 + '10 best selected features' + '-'*20)



selected_values = X.columns.values[mask_kbest]

print('\ncols of dataset: ', creditcard_data.columns)
print('\nselected cols: ',selected_values)

cols = [i for i in selected_values]

X_select = X_kbest
X_select = pd.DataFrame(data = X_select[0:,0:],
                        index=[i for i in range(X_select.shape[0])],
                        columns=cols)

print(X_select)
print('Type of X: ',type(X))
print('Type of X_select: ',type(X_select))

#split
X_train_base, X_test, Y_train_base, Y_test = train_test_split(X_select,Y, test_size=.25,
                                                              random_state=42)

print('Shape of X_train_base: ', X_train_base.shape)
print('Shape of Y_train_base: ', Y_train_base.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of Y_test: ', Y_test.shape)


#SMOTE
print('-'*15 + 'SMOTE' + '-'*15)
os = SMOTE(random_state=0)

#identify outliers in the training dataset
lr = [0.011]

rm_num = []

kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=2)

noised_num = {}
for k in lr:
    noised_num[k] =0

#------------------- MODELS DEFINITION
# putting all the model names, model classes and the used columns in a dictionary
# Before preprocessing data
models = {#'Logistic': [logit, cols],
          #'DecisionTree': [decision_tree, cols],
          #'KNN': [knn, cols],
          #'RandomForest': [rf, cols],
          #'NaiveBayes': [nb, cols],
          #'LGBM': [lgbmc, cols],
          #'XGBoost': [xgc, cols],
          #'Gaussian': [gpc, cols],
          #'AdaBoost': [adac, cols],
          #'GradientBoost': [gbc, cols],
          #'LDA': [lda, cols],
          #'QDA': [qda, cols],
          #'MLP': [mlp, cols],
          #'Bagging': [bgc, cols],
          'SVM_linear': [svc_lin, cols],
          'SVM_rbf': [svc_rbf, cols]
          }

# Applying SMOTE (smote)
models_smote = {#'Logistic_SMOTE': [logit_smote, cols],
                #'DecisionTree_SMOTE': [decision_tree_smote, cols],
                #'KNN_SMOTE': [knn_smote, cols],
                #'RandomForest_SMOTE': [rf_smote, cols],
                #'NaiveBayes_SMOTE': [nb_smote, cols],
                #'LGBM_SMOTE': [lgbmc_smote, cols],
                #'XGBoost_SMOTE': [xgc_smote, cols],
                #'Gaussian_SMOTE': [gpc_smote, cols],
                #'AdaBoost_SMOTE': [adac_smote, cols],
                #'GradientBoost_SMOTE': [gbc_smote, cols],
                #'LDA_SMOTE': [lda_smote, cols],
                #'QDA_SMOTE': [qda_smote, cols],
                #'MLP_SMOTE': [mlp_smote, cols],
                #'Bagging_SMOTE': [bgc_smote, cols],
                'SVM_linear_SMOTE': [svc_lin_smote, cols],
                'SVM_rbf_SMOTE': [svc_rbf_smote, cols]
                }

#Applying after removing outliers
models_routliers = {#'Logistic_routliers': [logit_routliers, cols],
                    #'DecisionTree_routliers': [decision_tree_routliers, cols],
                    #'KNN_routliers': [knn_routliers, cols],
                    #'RandomForest_routliers': [rf_routliers, cols],
                    #'NaiveBayes_routliers': [nb_routliers, cols],
                    #'LGBM_routliers': [lgbmc_routliers, cols],
                    #'XGBoost_routliers': [xgc_routliers, cols],
                    #'Gaussian_routliers': [gpc_routliers, cols],
                    #'AdaBoost_routliers': [adac_routliers, cols],
                    #'GradientBoost_routliers': [gbc_routliers, cols],
                    #'LDA_routliers': [lda_routliers, cols],
                    #'QDA_routliers': [qda_routliers, cols],
                    #'MLP_routliers': [mlp_routliers, cols],
                    #'Bagging_routliers': [bgc_routliers, cols],
                    'SVM_linear_routliers': [svc_lin_routliers, cols],
                    'SVM_rbf_routliers': [svc_rbf_routliers, cols]
                    }

# Applying Random Undersampling after Over-sampling SMOTE (rus)
models_rus = {#'Logistic_RUS': [logit_rus, cols],
              #'DecisionTree_RUS': [decision_tree_rus, cols],
              #'KNN_RUS': [knn_rus, cols],
              #'RandomForest_RUS': [rf_rus, cols],
              #'NaiveBayes_RUS': [nb_rus, cols],
              #'LGBM_RUS': [lgbmc_rus, cols],
              #'XGBoost_RUS': [xgc_rus, cols],
              #'Gaussian_RUS': [gpc_rus, cols],
              #'AdaBoost_RUS': [adac_rus, cols],
              #'GradientBoost_RUS': [gbc_rus, cols],
              #'LDA_RUS': [lda_rus, cols],
              #'QDA_RUS': [qda_rus, cols],
              #'MLP_RUS': [mlp_rus, cols],
              #'Bagging_RUS': [bgc_rus, cols],
              'SVM_linear_RUS': [svc_lin_rus, cols],
              'SVM_rbf_RUS': [svc_rbf_rus, cols]
              }

for r in lr:
    id_fold = 0
    for train_ix, test_ix in kfold.split(X_select,Y):
        X_train_base = X_select.iloc[train_ix]
        Y_train_base = Y.iloc[train_ix]
        X_test = X_select.iloc[test_ix]
        Y_test = Y.iloc[test_ix]



        iso = IsolationForest(contamination=r)
        yhat = iso.fit_predict(X_train_base)

        mask = yhat != -1

        rm_num.append(sum([1 for i in yhat if i == -1]))
        noised_num[r] += sum([1 for i in yhat if i == -1])
        X_train, Y_train = X_train_base.iloc[mask, :], Y_train_base.iloc[mask]
        #X_train, Y_train = X_train_base[mask, :], Y_train_base[mask]

        id_fold += 1

        print('-' * 30)
        print('          Fold - {}          '.format(id_fold))
        print('-' * 30)

        #### smote
        print('='*10 + '> resampling - SMOTE')
        os_smote_x, os_smote_y = os.fit_resample(X_train_base, Y_train_base)
        x_sm_base = pd.DataFrame(data = os_smote_x, columns=cols)
        y_sm_base = pd.DataFrame(data = os_smote_y, columns=target_col)
        '''
                    print('Fold: ', id_fold)
                    print('Shape of X_train_base: ', X_train_base.shape)
                    print('Shape of Y_train_base: ', Y_train_base.shape)
                    print('Shape of X_test: ', X_test.shape)
                    print('Shape of Y_test: ', Y_test.shape)
                    print('Shape of X_train: ', X_train.shape)
                    print('Shape of Y_train: ', Y_train.shape)
                    print('Shape of x_sm_base: ', x_sm_base.shape)
                    print('Shape of y_sm_base: ', y_sm_base.shape)
        '''

        #### remove noise
        print('=' * 10 + '> remove noise')
        smote_x, smote_y = os.fit_resample(X_train, Y_train)
        x_sm_base_r = pd.DataFrame(data = smote_x, columns=cols)
        y_sm_base_r = pd.DataFrame(data=smote_y, columns = target_col)

        #### random undersampling after smote and remove noise
        print('=' * 10 + '> Random undersampling')
        ru = RandomUnderSampler(random_state=0)
        ru_x, ru_y = ru.fit_resample(x_sm_base_r, y_sm_base_r)
        x_rus = pd.DataFrame(data= ru_x, columns=cols)
        y_rus = pd.DataFrame(data = ru_y, columns=target_col)

        # output for all models
        print('='*10 + '> Out put for all models')
        model_performances_train_baseline = pd.DataFrame()
        model_performances_train_smote = pd.DataFrame()
        model_performances_train_removeoutliers = pd.DataFrame()
        model_performances_train_rus = pd.DataFrame()

        labels = ['Not Fraud', 'Fraud']
        kinds = ['BASELINE', 'SMOTE', 'REMOVE_OUTLIERS', 'Random_Undersampling']

        for kind in kinds:
            print('#' * 25)
            print('# {} - Fold: {} '.format(kind, id_fold))
            print('#' * 25)

            if kind == 'BASELINE':
                for name in models:
                    #print('-'*10, name)
                    print('=*' * 20 + '\n||         ' + name + '       \n' + '=*' * 20)
                    model_performances_train_baseline = model_performances_train_baseline.append(model_report(models[name][0],
                                                                                                              X_train_base[models[name][1]],
                                                                                                              X_test[models[name][1]],
                                                                                                              Y_train_base, Y_test, name,
                                                                                                              'BASELINE', id_fold, 'credit'),
                                                                                                 ignore_index=True)
                ########################################
                # MODEL PERFORMANCES TRAIN             #
                ########################################
                table_train = ff.create_table(np.round(model_performances_train_baseline, 4))
                table_train.show()
                filename_ = 'results_fig/credit_Metrics_Table_{}{}.html'.format(kind, str(id_fold))
                py.iplot(table_train, filename=filename_)

                print('########################################')
                print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit_baseline"))
                print('########################################')
                confmatplot(modeldict=models, df_train=[X_train, X_train_base], df_test=X_test,
                            target_train=[Y_train, Y_train_base], target_test=Y_test, figcolnumber=3, kind=kind,
                            dataset_name='credit', fold=id_fold, labels=labels)

                print('########################################')
                print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit_baseline"))
                print('########################################')
                rocplot(modeldict=models, df_train=[X_train, X_train_base], df_test=X_test,
                        target_train=[Y_train, Y_train_base], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)

                print('########################################')
                print('# Precision recall curves------- {}# - model: {}#'.format(kind, "models_credit_baseline"))
                print('########################################')
                prcplot(modeldict=models, df_train=[X_train, X_train_base], df_test=X_test,
                        target_train=[Y_train, Y_train_base], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)
            elif kind == 'SMOTE':

                for name in models_smote:
                    #print('-' * 10, name)
                    print('=*' * 20 + '\n||         ' + name + '       \n' + '=*' * 20)
                    model_performances_train_smote = model_performances_train_smote.append(model_report(models_smote[name][0],
                                                                                                        x_sm_base[models_smote[name][1]],
                                                                                                        X_test[models_smote[name][1]],
                                                                                                        y_sm_base, Y_test, name,
                                                                                                        'SMOTE', id_fold, 'credit'),
                                                                                           ignore_index=True)
                ########################################
                # MODEL PERFORMANCES TRAIN             #
                ########################################
                table_train = ff.create_table(np.round(model_performances_train_smote, 4))
                table_train.show()
                filename_ = 'results_fig/credit_Metrics_Table_{}{}.html'.format(kind, str(id_fold))
                py.iplot(table_train, filename=filename_)

                print('########################################')
                print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit_smote"))
                print('########################################')
                confmatplot(modeldict=models_smote, df_train=[X_train, x_sm_base], df_test=X_test,
                            target_train=[Y_train, y_sm_base], target_test=Y_test, figcolnumber=3, kind=kind,
                            dataset_name='credit', fold=id_fold, labels=labels)

                print('########################################')
                print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit"))
                print('########################################')
                rocplot(modeldict=models_smote, df_train=[X_train, x_sm_base], df_test=X_test,
                        target_train=[Y_train, y_sm_base], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)

                print('########################################')
                print('# Precision recall curves------- {}# - model: {}#'.format(kind, "models_credit"))
                print('########################################')
                prcplot(modeldict=models_smote, df_train=[X_train, x_sm_base], df_test=X_test,
                        target_train=[Y_train, y_sm_base], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)

            elif kind == 'REMOVE_OUTLIERS':
                for name in models_routliers:
                    #print('-' * 10, name)
                    print('=*' * 20 + '\n||         ' + name + '       \n' + '=*' * 20)
                    model_performances_train_removeoutliers = model_performances_train_removeoutliers.append(model_report(models_routliers[name][0],
                                                                                                                          x_sm_base_r[models_routliers[name][1]],
                                                                                                                          X_test[models_routliers[name][1]],
                                                                                                                          y_sm_base_r, Y_test, name,
                                                                                                                          'REMOVE_OUTLIERS', id_fold,
                                                                                                                          'credit'),
                                                                                                             ignore_index=True)

                ########################################
                # MODEL PERFORMANCES TRAIN             #
                ########################################
                table_train = ff.create_table(np.round(model_performances_train_removeoutliers, 4))
                table_train.show()
                filename_ = 'results_fig/credit_Metrics_Table_{}{}.html'.format(kind, str(id_fold))
                py.iplot(table_train, filename=filename_)

                print('########################################')
                print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit_remove_outliers"))
                print('########################################')
                confmatplot(modeldict=models_routliers, df_train=[X_train, x_sm_base_r], df_test=X_test,
                            target_train=[Y_train, y_sm_base_r], target_test=Y_test, figcolnumber=3, kind=kind,
                            dataset_name='credit', fold=id_fold, labels=labels)

                print('########################################')
                print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit"))
                print('########################################')
                rocplot(modeldict=models_routliers, df_train=[X_train, x_sm_base_r], df_test=X_test,
                        target_train=[Y_train, y_sm_base_r], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)

                print('########################################')
                print('# Precision recall curves------- {}# - model: {}#'.format(kind, "models_credit"))
                print('########################################')
                prcplot(modeldict=models_routliers, df_train=[X_train, x_sm_base_r], df_test=X_test,
                        target_train=[Y_train, y_sm_base_r], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)


            elif kind == 'Random_Undersampling':
                for name in models_rus:
                    #print('-' * 10, name)
                    print('=*' * 20 + '\n||         ' + name + '       \n' + '=*' * 20)
                    model_performances_train_rus = model_performances_train_rus.append(model_report(models_rus[name][0],
                                                                                                    x_rus[models_rus[name][1]],
                                                                                                    X_test[models_rus[name][1]],
                                                                                                    y_rus, Y_test, name,
                                                                                                    'Random_Undersampling',
                                                                                                    id_fold,
                                                                                                    'credit'),
                                                                                       ignore_index=True)

                ########################################
                # MODEL PERFORMANCES TRAIN             #
                ########################################
                table_train = ff.create_table(np.round(model_performances_train_rus, 4))
                table_train.show()
                filename_ = 'results_fig/credit_Metrics_Table_{}{}.html'.format(kind, str(id_fold))
                py.iplot(table_train, filename=filename_)

                print('########################################')
                print('# CONFUSION MATRIX------- {} - model: {}#'.format(kind, "models_credit_random_undersampling"))
                print('########################################')
                confmatplot(modeldict=models_rus, df_train=[X_train, x_rus], df_test=X_test,
                            target_train=[Y_train, y_rus], target_test=Y_test, figcolnumber=3, kind=kind,
                            dataset_name='credit', fold=id_fold, labels=labels)

                print('########################################')
                print('# ROC - Curves for models------- {} - model: {}#'.format(kind, "models_credit"))
                print('########################################')
                rocplot(modeldict=models_rus, df_train=[X_train, x_rus], df_test=X_test,
                        target_train=[Y_train, y_rus], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)

                print('########################################')
                print('# Precision recall curves------- {}# - model: {}#'.format(kind, "models_credit"))
                print('########################################')
                prcplot(modeldict=models_rus, df_train=[X_train, x_rus], df_test=X_test,
                        target_train=[Y_train, y_rus], target_test=Y_test, figcolnumber=3, kind=kind,
                        dataset_name='credit', fold=id_fold)
