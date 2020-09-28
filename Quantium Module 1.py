#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[6]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import missingno

# dates
import datetime
from matplotlib.dates import DateFormatter

# text analysis
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist as fdist
import re

# statistical analysis
from scipy.stats import ttest_ind

# warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pwd


# # Load data

# In[7]:


customerData = pd.read_csv("C:/Users/Jason Chong/Documents/Inside Sherpa Quantium/QVI_purchase_behaviour.csv")
transactionData = pd.read_csv("C:/Users/Jason Chong/Documents/Inside Sherpa Quantium/QVI_transaction_data.csv")


# # Transaction data

# In[8]:


transactionData.head()


# In[10]:


transactionData.shape


# In[11]:


transactionData['TXN_ID'].nunique()

# Comment: 'TXN_ID' is not unique to each row which means a customer can buy multiple brands of chips in a single trasaction
# Each row of the data corresponds to purchase of a single brand of chips and a customer can buy multiple brands in one single transaction


# In[95]:


# For example, look for duplicated 'TXN_ID'

transactionData[transactionData.duplicated(['TXN_ID'])].head()


# In[96]:


# Let's have a look at 'TXN_ID' 48887

transactionData.loc[transactionData['TXN_ID'] == 48887, :]


# In[12]:


transactionData.info()


# In[13]:


# Plot graph of missing values for 'transactionData'

missingno.matrix(transactionData)

# Comment: no missing numbers in transaction data 


# In[9]:


# Now let's explore the features in both dataset starting with 'transactionData'

list(transactionData.columns)


# In[10]:


transactionData['DATE'].head()


# In[14]:


# 'Date' is not in the right format

# Function that converts Excel integer into yyyy-mm-dd format
def xlseriesdate_to_datetime(xlserialdate):
    excel_anchor = datetime.datetime(1900, 1, 1)
    if(xlserialdate < 60):
        delta_in_days = datetime.timedelta(days = (xlserialdate - 1))
    else:
        delta_in_days = datetime.timedelta(days = (xlserialdate - 2))
    converted_date = excel_anchor + delta_in_days
    return converted_date


# In[15]:


# Apply function to 'DATE' feature in 'transactionData' dataset

transactionData['DATE'] = transactionData['DATE'].apply(xlseriesdate_to_datetime)


# In[16]:


# Check the first 5 rows of the new feature

transactionData['DATE'].head()


# In[17]:


# Date is now in the right format

transactionData.head()


# In[15]:


# Let's move on to 'PROD_NAME' feature

transactionData['PROD_NAME'].head()


# In[18]:


# Extract weights out of 'PROD_NAME'

transactionData['PACK_SIZE'] = transactionData['PROD_NAME'].str.extract("(\d+)")
transactionData['PACK_SIZE'] = pd.to_numeric(transactionData['PACK_SIZE'])
transactionData.head()


# In[19]:


# Create text cleaning function for 'PROD_NAME' feature
def clean_text(text):
    text = re.sub('[&/]', ' ', text) # remove special characters '&' and '/'
    text = re.sub('\d\w*', ' ', text) # remove product weights
    return text

# Apply text cleaning function to 'PROD_NAME' column
transactionData['PROD_NAME'] = transactionData['PROD_NAME'].apply(clean_text)


# In[20]:


# Create one giant string and apply 'word_tokenize' to separate the words

cleanProdName = transactionData['PROD_NAME']
string = ''.join(cleanProdName)
prodWord = word_tokenize(string)


# In[21]:


# Apply 'fdist' function which computes the frequency of each token and put it into a dataframe

wordFrequency = fdist(prodWord)
freq_df = pd.DataFrame(list(wordFrequency.items()), columns = ["Word", "Frequency"]).sort_values(by = 'Frequency', ascending = False)


# In[22]:


# Let's see the top 5 most frequent words

freq_df.head()


# In[23]:


# Drop rows with 'salsa' in 'PROD_NAME'

transactionData['PROD_NAME'] = transactionData['PROD_NAME'].apply(lambda x: x.lower())
transactionData = transactionData[~transactionData['PROD_NAME'].str.contains("salsa")]
transactionData['PROD_NAME'] = transactionData['PROD_NAME'].apply(lambda x: x.title())


# In[24]:


# Let's have a look at our data table again

transactionData.head()


# In[25]:


# We shall explore 'PROD_QTY' and 'TOT_SALES' feature next

transactionData['PROD_QTY'].value_counts()

# Max of 200 looks odd


# In[26]:


# We have two occurrences of 200 in the dataset
# Let's explore further

transactionData.loc[transactionData['PROD_QTY'] == 200, :]


# In[27]:


# Both these transactions have been made by the same person at the same store
# Let's see all the transactions this person has made by tracking his loyalty card number '226000'

transactionData.loc[transactionData['LYLTY_CARD_NBR'] == 226000, :]


# In[28]:


# This person only made two transactions over the entire year so unlikely to be a retail customer 
# He or she is most likely purchasing for commercial purposes
# Safe to drop these this customer in both 'transactionData' and 'customerData' dataset

transactionData.drop(transactionData.index[transactionData['LYLTY_CARD_NBR'] == 226000], inplace = True)
customerData.drop(customerData.index[customerData['LYLTY_CARD_NBR'] == 226000], inplace = True)


# In[29]:


# Make sure it has been dropped 

transactionData.loc[transactionData['LYLTY_CARD_NBR'] == 226000]


# In[30]:


# Now let's examine the number of transactions over time to see if there are any obvious data issues e.g. missing data

transactionData['DATE'].nunique()


# In[31]:


# Look for the missing date 
# Turns out that it was Christmas Day so it makes sense because most retail stores are closed on that day

pd.date_range(start = '2018-07-01', end = '2019-06-30').difference(transactionData['DATE'])


# In[32]:


# Create a new dataframe which contains the total sale for each date

a = pd.pivot_table(transactionData, values = 'TOT_SALES', index = 'DATE', aggfunc = 'sum')
a.head()


# In[33]:


b = pd.DataFrame(index = pd.date_range(start = '2018-07-01', end = '2019-06-30'))
b['TOT_SALES'] = 0
b


# In[34]:


c = a+b
c.fillna(0, inplace = True)


# In[35]:


c.head()


# In[36]:


c.index.name = 'Date'
c.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
c.head() 


# In[38]:


timeline = c.index
graph = c['Total Sales']

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(timeline, graph)

date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.title('Total Sales from July 2018 to June 2019')
plt.xlabel('Time')
plt.ylabel('Total Sales')

plt.show()

# Comment: We can see that sales spike up during the December month and zero sale on Christmas


# In[39]:


# Confirm the date where sales count equals to zero

c[c['Total Sales'] == 0]

# It is indeed Christmas Day


# In[40]:


# Let's look at the December month only

c_december = c[(c.index < "2019-01-01") & (c.index > "2018-11-30")]
c_december.head()


# In[41]:


plt.figure(figsize = (15, 5))
plt.plot(c_december)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Total Sales in December')


# In[39]:


# Reset index for 'c_december'

c_december.reset_index(drop = True, inplace = True)
c_december.head()


# In[40]:


# Relabel 'Date'

c_december['Date'] = c_december.index + 1
c_december.head()


# In[41]:


plt.figure(figsize = (15,5))
sns.barplot(x = 'Date', y ='Total Sales', data = c_december)

# Comment: No sales on Christmas Day (2018-12-25)


# In[42]:


# Now that we are satisfied that the data no longer has outliers
# We can move on to creating other features
# At the beginning, we have already created a 'pack_size' feature
# Let's have a look again

transactionData['PACK_SIZE'].head()


# In[43]:


transactionData['PACK_SIZE'].unique()

# Comment: the largest size is 380g and the smallest size is 70g which seems reasonable


# In[44]:


# Check the distribution of 'PACK_SIZE'

transactionData['PACK_SIZE'].hist()


# In[45]:


# Extract brand name from 'PROD_NAME' 
# Create a new column under 'TransactionData' called 'brand'

part = transactionData['PROD_NAME'].str.partition()
transactionData['BRAND'] = part[0]
transactionData.head()


# In[46]:


transactionData['BRAND'].unique()


# In[47]:


# It looks like there are duplicates of the same brand e.g. 'ww' and 'woolworths', 'red' and 'rrd', 'natural' and 'ncc', 
# 'infuzions' and 'infzns', 'snbts' and 'sunbites', 'grain' and 'grnwves', 'smiths' and 'smith', 'doritos' and 'dorito'
# Let's rename them for consistency

transactionData['BRAND'].replace('Ncc', 'Natural', inplace = True)
transactionData['BRAND'].replace('Ccs', 'CCS', inplace = True)
transactionData['BRAND'].replace('Smith', 'Smiths', inplace = True)
transactionData['BRAND'].replace(['Grain', 'Grnwves'], 'Grainwaves', inplace = True)
transactionData['BRAND'].replace('Dorito', 'Doritos', inplace = True)
transactionData['BRAND'].replace('Ww', 'Woolworths', inplace = True)
transactionData['BRAND'].replace('Infzns', 'Infuzions', inplace = True)
transactionData['BRAND'].replace(['Red', 'Rrd'], 'Red Rock Deli', inplace = True)
transactionData['BRAND'].replace('Snbts', 'Sunbites', inplace = True)

transactionData['BRAND'].unique()


# In[48]:


# Which brand had the most sales?

transactionData.groupby('BRAND').TOT_SALES.sum().sort_values(ascending = False)


# # Customer Data

# In[49]:


# Let's move on to 'customerData' dataset now

list(customerData.columns)


# In[50]:


customerData.head()


# In[51]:


missingno.matrix(customerData)


# In[51]:


len(customerData)


# In[52]:


customerData['LYLTY_CARD_NBR'].nunique()

# Comment: 'LYLTY_CARD_NBR' is unique to each row


# In[53]:


customerData['LIFESTAGE'].nunique()

# Comment: 7 unique lifestages of customers


# In[54]:


# Let's see what those lifestages are

customerData['LIFESTAGE'].unique()


# In[55]:


# Counts for each lifestages

customerData['LIFESTAGE'].value_counts().sort_values(ascending = False)


# In[56]:


sns.countplot(y = customerData['LIFESTAGE'], order = customerData['LIFESTAGE'].value_counts().index)


# In[57]:


# What about the 'PREMIUM_CUSTOMER' column

customerData['PREMIUM_CUSTOMER'].nunique()


# In[58]:


# Counts for each 'PREMIUM_CUSTOMER'

customerData['PREMIUM_CUSTOMER'].value_counts().sort_values(ascending = False)

# Comment: Mainstream has the highest count, followed by budget and finally premium


# In[59]:


# Visualise 'PREMIUM_CUSTOMER'

sns.countplot(y = customerData['PREMIUM_CUSTOMER'])


# In[60]:


# Now let's merge the two datasets together
# Before we do, examine the shape

transactionData.shape


# In[61]:


customerData.shape


# In[62]:


combineData = pd.merge(transactionData, customerData)
combineData.shape


# In[63]:


# The two datasets are joined together via the column 'LYLTY_CARD_NBR'

combineData.head()


# In[64]:


# Check for null values

combineData.isnull().sum()


# # Data analysis on customer segments
# 
# Now that our data is ready for analysis, we can define some metrics of interest to the client:
# 
# - Who spends the most on chips, describing customers by lifestage and how premium their general purchasing behaviour is
# - How many customers are in each segment
# - How many chips are bought per customer by segment
# - What is the average chip price by customer segment
#     

# In[65]:


# Total sales by 'PREMIUM_CUSTOMER' and 'LIFESTAGE'

sales = pd.DataFrame(combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum())
sales.rename(columns = {'TOT_SALES': 'Total Sales'}, inplace = True)
sales.sort_values(by = 'Total Sales', ascending = False, inplace = True)
sales


# In[66]:


# Visualise
salesPlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum())
salesPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (10, 5), title = 'Total Sales by Customer Segment')

# Comment: Sales are coming from budget older families, mainstream young singles/couples and mainstream retirees


# In[70]:


# Let's see if the higher sales are due to there being more customers who buy chips
# Number of customers by 'PREMIUM_CUSTOMER' and 'LIFESTAGE'
# Remember to take unique 'LYLTY_CARD_NBR'

customers = pd.DataFrame(combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique())
customers.rename(columns = {'LYLTY_CARD_NBR': 'Number of Customers'}, inplace = True)
customers.sort_values(by = 'Number of Customers', ascending = False).head(10)


# In[69]:


# Visualise
customersPlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
customersPlot.unstack().plot(kind = 'bar', stacked = True, figsize = (10, 5), title = 'Number of Customers by Customer Segment')

# Comment: There are more mainstream young singles/couples and retirees. This contributes to to more chips sales in these
# segments however this is not the major driver for the budget older families segment


# In[71]:


# Higher sales may also be driven by more units of chips being bought per customer
# Let's calculate the average units per customer by 'PREMIUM_CUSTOMER' and 'LIFESTAGE'
# Total quantity sold divided by unique customers

avg_units = combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum() / combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique()
avg_units = pd.DataFrame(avg_units, columns = {'Average Unit per Customer'})
avg_units.sort_values(by = 'Average Unit per Customer', ascending = False).head()


# In[72]:


# Visualise 
avgUnitsPlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum() / combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
avgUnitsPlot.unstack().plot(kind = 'bar', figsize = (10, 5), title = 'Average Unit by Customer Segment')

# Comment: Older families and young families buy more chips per customer


# In[73]:


# Let's also investigate the average price per unit chips bought for each customer segment as this is also a driver of total sales
# Total sales divided by total quantity purchased

# Average price per unit by 'PREMIUM_CUSTOMER' and 'LIFESTAGE'
avg_price = combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum() / combineData.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum()
avg_price = pd.DataFrame(avg_price, columns = {'Price per Unit'})
avg_price.sort_values(by = 'Price per Unit', ascending = False).head()


# In[74]:


# Visualise 
avgPricePlot = pd.DataFrame(combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum() / combineData.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum())
avgPricePlot.unstack().plot(kind = 'bar', figsize = (10, 5), title = 'Average Price by Customer Segment', ylim = (0, 6))

# Comment: Mainstream midage and young singles and couples are more willing to pay more per packet of chips compared to their 
# budget and premium counterparts. This may be due to premium shoppers being more likely to buy healthy snacks and when
# they do buy chips, it is mainly for entertainment purposes rather than their own consumption. This is also supported by
# there being fewer premium midage and young singles and couples buying chips compared to their mainstream counterparts


# In[75]:


# As this difference in average price per unit is not too large 
# Perform an independent t-test between mainstream vs non-mainstream midage and young singles/couples

# Create a new dataframe 'pricePerUnit'
pricePerUnit = combineData

# Create a new column under 'pricePerUnit' called 'PRICE'
pricePerUnit['PRICE'] = pricePerUnit['TOT_SALES'] / pricePerUnit['PROD_QTY']

# Let's have a look
pricePerUnit.head()


# In[76]:


# Let's group our data into mainstream and non-mainstream

mainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] == 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']
nonMainstream = pricePerUnit.loc[(pricePerUnit['PREMIUM_CUSTOMER'] != 'Mainstream') & ( (pricePerUnit['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') | (pricePerUnit['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES') ), 'PRICE']


# In[77]:


plt.figure(figsize = (10, 5))
plt.hist(mainstream, label = 'Mainstream')
plt.hist(nonMainstream, label = 'Premium & Budget')
plt.legend()
plt.xlabel('Price per Unit')


# In[78]:


# Let's have a look at their means

[np.mean(mainstream), np.mean(nonMainstream)]

# Mainstream has a higher average price per unit


# In[79]:


# Perform t-test 

ttest_ind(mainstream, nonMainstream)

# Comment: Mainstream price per unit is significantly higher than non-mainstream 


# In[80]:


# Deep dive into specific customer segment for insights
# We have found quite a few interesting insights that we can dive deeper into 
# For example, we might want to target customers segments that contribute the most to sales to retain them to further increase sales
# Let's examine mainstream young singles/couples against the rest of the cutomer segments to see if they prefer any particular brand of chips

target = combineData.loc[(combineData['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (combineData['PREMIUM_CUSTOMER'] == 'Mainstream'), :]
nonTarget = combineData.loc[(combineData['LIFESTAGE'] != 'YOUNG SINGLES/COUPLES' ) & (combineData['PREMIUM_CUSTOMER'] != 'Mainstream'), :]
target.head()


# In[81]:


# Affinity to brand

# Target Segment
targetBrand = target.loc[:, ['BRAND', 'PROD_QTY']]
targetSum = targetBrand['PROD_QTY'].sum()
targetBrand['Target Brand Affinity'] = targetBrand['PROD_QTY'] / targetSum
targetBrand = pd.DataFrame(targetBrand.groupby('BRAND')['Target Brand Affinity'].sum())

# Non-target segment
nonTargetBrand = nonTarget.loc[:, ['BRAND', 'PROD_QTY']]
nonTargetSum = nonTargetBrand['PROD_QTY'].sum()
nonTargetBrand['Non-Target Brand Affinity'] = nonTargetBrand['PROD_QTY'] / nonTargetSum
nonTargetBrand = pd.DataFrame(nonTargetBrand.groupby('BRAND')['Non-Target Brand Affinity'].sum())


# In[82]:


# Merge the two dataframes together

brand_proportions = pd.merge(targetBrand, nonTargetBrand, left_index = True, right_index = True)
brand_proportions.head()


# In[83]:


brand_proportions['Affinity to Brand'] = brand_proportions['Target Brand Affinity'] / brand_proportions['Non-Target Brand Affinity']
brand_proportions.sort_values(by = 'Affinity to Brand', ascending = False)


# In[ ]:


# Comment: Mainstream young singles/couples are more likely to purchase Tyrrells chips compared to other brands


# In[84]:


# Affinity to pack size

# Target Segment
targetSize = target.loc[:, ['PACK_SIZE', 'PROD_QTY']]
targetSum = targetSize['PROD_QTY'].sum()
targetSize['Target Pack Affinity'] = targetSize['PROD_QTY'] / targetSum
targetSize = pd.DataFrame(targetSize.groupby('PACK_SIZE')['Target Pack Affinity'].sum())

# Non-target segment
nonTargetSize = nonTarget.loc[:, ['PACK_SIZE', 'PROD_QTY']]
nonTargetSum = nonTargetSize['PROD_QTY'].sum()
nonTargetSize['Non-Target Pack Affinity'] = nonTargetSize['PROD_QTY'] / nonTargetSum
nonTargetSize = pd.DataFrame(nonTargetSize.groupby('PACK_SIZE')['Non-Target Pack Affinity'].sum())


# In[85]:


# Merge the two dataframes together

pack_proportions = pd.merge(targetSize, nonTargetSize, left_index = True, right_index = True)
pack_proportions.head()


# In[86]:


pack_proportions['Affinity to Pack'] = pack_proportions['Target Pack Affinity'] / pack_proportions['Non-Target Pack Affinity']
pack_proportions.sort_values(by = 'Affinity to Pack', ascending = False)


# In[ ]:


# Comment: It looks like mainstream singles/couples are more likely to purchase a 270g pack size compared to other pack sizes


# In[87]:


# Which brand offers 270g pack size

combineData.loc[combineData['PACK_SIZE'] == 270, :].head(10)


# In[88]:


# Is Twisties the only brand who sells 270g pack size

combineData.loc[combineData['PACK_SIZE'] == 270, 'BRAND'].unique()

# Twisties is the only brand that offers 270g pack size 


# # Conclusion
# 
# - Sales are highest for (Budget, OLDER FAMILIES), (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES)
# - We found that (Mainstream, YOUNG SINGLES/COUPLES) and (Mainstream, RETIREES) are mainly due to the fact that there are more customers in these segments
# - (Mainstream, YOUNG SINGLES/COUPLES) are more likely to pay more per packet of chips than their premium and budget counterparts
# - They are also more likely to purchase 'Tyrrells' and '270g' pack sizes than the rest of the population

# # Recommendation
# 
# The category manager may consider off-locating 'Tyrrells' chips in discretionary space near segments where young singles and couples frequent to increase the visibility and impulse behaviour
