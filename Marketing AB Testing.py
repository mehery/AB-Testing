import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/kaggle/input/marketing-ab-testing/marketing_AB.csv")


# We are trying to understand the data.

def check_df(dataframe, head=7):
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#################### Head ####################")
    print(dataframe.head(head))

check_df(df)


# Data Preparation

# We are deleting the variable that does not carry any information.
df.drop("Unnamed: 0", inplace=True, axis=1)

# We convert the true/false values to 1 and 0.
df["converted"] = np.where(df["converted"]==False, 0, 1)

df.head()


# We are looking at the mean purchase values of those who saw the advertisement and those who didn't.

df.groupby("test group")["converted"].mean()


# We are summing the purchase values separately for those who saw the ad and those who didn't see the ad.
# We assign these to new variables.

ad_converted_count = df.loc[df["test group"] == "ad", "converted"].sum()
psa_converted_count = df.loc[df["test group"] == "psa", "converted"].sum()


# We are calculating the p-value to determine the effect of seeing, the advertisement on the purchase for those who saw it versus those who didn't see it.

test_stat, pvalue = proportions_ztest(count=[ad_converted_count, psa_converted_count],
                                      nobs=[df.loc[df["test group"] == "ad", "converted"].shape[0],
                                            df.loc[df["test group"] == "psa", "converted"].shape[0]])

# count = success count
# nobs = the total number of observations
# Thus, we obtain the ratio.


print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# Since p < 0.05, H0 is rejected, meaning there is a statistically significant difference between the two groups.¶
# So the advertisement has an effect on the purchase, the advertisement is successful.
