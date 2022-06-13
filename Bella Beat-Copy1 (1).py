#!/usr/bin/env python
# coding: utf-8

# # BellaBeat Exploratory Analysis
# 
# ### Cleaning, Code, and Comments by Cristi Marascio

# ## Executive Summary
# 
# This EDA will focus on the Time product line as it is most like the FitBit that was used to extract the data. 
# 1. The more features on the device i.e. Activity, sleep, and weight trackers the user engaged with the more consistently they used the product. 
# 2. Since the Fitbit and Time are similar in use and demographic we can apply trends noted in use to help market Time. 
# 3. We see a reduction in use among most of the users as time progresses. Further study is needed around the 20-30 day from purchase mark to see why use falls off, but gathering that data can help us create a marketing strategy to increase engagement and create a more loyal customer. 

# ## Ask
# 1. What are some trends in smart device usage?
# 2. How could these trends apply to Bellabeat customers?
# 3. How could these trends help influence Bellabeat marketing strategy?

# #### Import the Libraries to be used

# In[3]:


# import libraries for data manipulation
import numpy as np
import pandas as pd

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# ## Clean the data 
# 
# 
# Does the Data ROCC?
# 
# R - This data is the usage of a fitbit device for 33 individuals for 31 days. 
# 
# O - The data come from a single source on Kaggle and is cited above.
# 
# C - The comprehension is limited since the data is a small sample size for a short duration. This limits the scope of the analysis. 
# 
# C - Current the data is not current and should not be used to predict trends, but is useful for the the descriptive study of how individuals use the device. 
# 
# C - See citation above to access the original dataset. 
# 
# ##### Data was intially cleaned using SQL. The tables Weight Log, Sleep Log, and Daily Activity were created from the original dataset found at https://www.kaggle.com/datasets/arashnic/fitbit

# #### Creates a table for Sleep info and calculates Percent_Sleep column
# 
# 
# SELECT Cast(Id as String) as Id_str
# 
# ,Cast(SleepDay as Date) as Day
# 
# ,TotalMinutesAsleep
# 
# ,TotalTimeInBed
# 
# ,Round((TotalMinutesAsleep/TotalTimeInBed)*100, 2) As Percent_Sleep
# 
# FROM `fitness-tracker-349421.Daily.dSleep`

# #### Extracts and cleans colum from 'weightLoginfo_merged' table 
# I only pulled pounds since that is the common unit of weight in the US
# 
# 
# SELECT
# 
# Cast (Id as String) as Id_str
# 
# , Day
# 
# , Round(WeightPounds, 2) as Lbs  
# 
# , IsManualReport
# 
# FROM `fitness-tracker-349421.Daily.WeightLog`

# #### Extracts and cleans colums from 'dailyActivity_merged' table  
# 
# 
# SELECT Cast(Id as string) AS ID_str
# 
# ,ActivityDate as Day
# 
# ,TotalSteps
# 
# ,Round(TotalDistance, 2) as D_total
# 
# ,Round(VeryActiveDistance, 2)as D_VeryActive
# 
# ,Round(ModeratelyActiveDistance, 2) as D_ModActive
# 
# ,Round(LightActiveDistance, 2) as D_LightActive
# 
# ,VeryActiveMinutes
# 
# ,FairlyActiveMinutes
# 
# ,LightlyActiveMinutes
# 
# ,SedentaryMinutes
# 
# ,Calories
# 
# FROM `fitness-tracker-349421.Daily.dAct` as dAct_C

# #### Creates a column to describe the amount of activity per day 
# ##### Recommended daily activity is 30 min/day or ~2% of your day
# 
# WITH Percent_Act as
# 
# (
# 
# SELECT
# 
# Id_str
# 
# ,Day
# 
# ,TotalSteps
# 
# ,D_total
# 
# ,(VeryActiveMinutes + FairlyActiveMinutes) as ActiveTime
# 
# ,Round((((VeryActiveMinutes + FairlyActiveMinutes)/1440)*100), 2) as PercentActive
# 
# 
# 
# FROM `fitness-tracker-349421.Daily.dAct_C` 
# 
# )
# 
# SELECT *
# 
# FROM Percent_Act

# ### Import the data to be analyzed

# In[4]:


# Import data to be analyzed: Weight Log
w=pd.read_csv('weight.csv')
w.info()


# In[5]:


# Import data to be analyzed: Quality of Sleep Log
s=pd.read_csv('Sleep.csv')
s.info()


# In[6]:


# Import data to be analyzed: Daily Activity Log
d=pd.read_csv('Percent_Act.csv')
d.info()


# In[8]:


#change day from object to string
d['Day'] = d['Day'].astype(str)
s['Day'] = s['Day'].astype(str)
w['Day'] = w['Day'].astype(str)


# In[9]:


# change Id_str datatype to string
d['Id_str'] = d['Id_str'].astype(str)
s['Id_str'] = s['Id_str'].astype(str)
w['Id_str'] = w['Id_str'].astype(str)


# In[10]:


#change Day data typ to date for all Logs
d['Day'] =  pd.to_datetime(d['Day'], format='%Y-%m-%d')
s['Day'] =  pd.to_datetime(s['Day'], format='%Y-%m-%d')
w['Day'] =  pd.to_datetime(w['Day'], format='%Y-%m-%d')

# Used to check success
d.info()
w.info()
s.info()


# In[11]:


# Describe each of the values and spread for numeric values in each table. 
d.describe(include='all', datetime_is_numeric=True).T


# In[12]:


s.describe(include='all', datetime_is_numeric=True ).T


# In[13]:


w.describe(include='all', datetime_is_numeric=True).T


# In[14]:


# Make a table that has all daily activity and sleep information available for those that use the sleep log function
ta = pd.merge(d, s, on=['Id_str', 'Day'] )
ta.head()


# In[15]:


ta.info()


# In[33]:


#Makes a table taht has all daily activity and weight information for those that use the weight tracker. 
wa = pd.merge(d, w, on=['Id_str', 'Day'] )
wa.head()


# ## Visualize the Data

# In[16]:


# Count the unique users in the dataframe 'PercentAct'
d['Id_str'].nunique()


# In[17]:


# Count the days included in the dataframe 'PercentAct'
d['Day'].nunique()


# #### The data has 33 unique users that tracked data for 31 days or an entire month. 

# In[18]:


# Plot the heatmap 
col_list = ['TotalSteps', 'ActiveTime', 'Percent_sleep']
plt.figure(figsize=(15, 7))
sns.heatmap(ta[col_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# #### Total steps and Active time seem to be the only numeric valuse that have a positive correlation. This makes sense since we expect you time active go up the farther or more steps you take. 

# In[19]:


# Create a dataframes that will help compare beginning of the month and end of month to evaluate good habits. 

begin_month = d[d['Day']< '2016-05-01']
end_month = d[d['Day']>= '2016-05-01']


end_month.head()


# In[20]:


begin_month.head()


# In[21]:


plt.figure(figsize=(25, 13))
a=sns.boxplot(x='Id_str', y = 'PercentActive', data=d, palette='bright')
plt.xticks(rotation = 45)
a.set_title("Percent Activity Spread for each user ID.", fontsize=30)
plt.show() ##plots the ID vs Percent Activity 


# In[22]:


b_count = begin_month[begin_month['ActiveTime']>30]

b_count = b_count.groupby(['Id_str'], as_index = False)['ActiveTime'].count()

b_count.head()
# Creates a table with count of days with more than 30 min of activity for the beginning of the month


# In[23]:


b_count.shape[0] #counts users with more than 30 min activity per day in the beginning of the month


# In[24]:


e_count = end_month[end_month['ActiveTime']>30]

e_count = e_count.groupby(['Id_str'], as_index = False)['ActiveTime'].count()

e_count.head()
# Creates a table with count of days with more than 30 min of activity for the end of the month


# In[25]:


e_count.shape[0]  #counts users with more than 30 min activity per day in the beginning of the month


# #### We had less users get the minumum suggested amount of activity at the end of the month than at the beginning. This could be due to loss of motivation. 

# In[26]:


step_count = d.groupby('Id_str').agg('median').reset_index()
step_count


# In[55]:


plt.figure(figsize=(25, 13))
d=sns.boxplot(x='Id_str', y = 'TotalSteps', data=d, palette='bright')
plt.xticks(rotation = 45)
d.set_title("Total Steps spread for each user.", fontsize=30)
plt.show()


# In[26]:


all_count = b_count.merge(e_count, how = 'outer', on = 'Id_str', suffixes=('_b', '_e')).fillna(0)
all_count


# In[27]:


ac=pd.melt(all_count, id_vars = 'Id_str', value_vars = ['ActiveTime_b', 'ActiveTime_e'])
ac.head()


# In[28]:


ac['Id_str'].nunique() 


# In[29]:


plt.gcf().set_size_inches(10,10)
b=sns.catplot(x='Id_str', y = 'value', data=ac, palette='bright', hue = 'variable', kind='bar')
plt.xticks(rotation = 90)
#b.set_title("Count of days that had more than 30 min of activty for each user.", fontsize=30)
plt.show(); ##plots the ID vs Percent Activity 


# #### The number of active days for users that were consistent went down as the month progressed. We can assume this is due to lack of motivation, or injury, or the novelty effect, or perhaps some other reason. Sending out surveys for users around the 20-30 mark would yield more information as to why we see a reduction in use.  

# In[30]:


no_act_b = begin_month[begin_month['TotalSteps']==0]
no_act_b.head() # creates table that counts inactive days for each user in the beginning of the month. 


# In[31]:


no_act_b = no_act_b.groupby(['Id_str'], as_index = True)['TotalSteps'].size()
no_act_b


# In[32]:


no_act_e = end_month[end_month['TotalSteps']==0]
no_act_e.head() # creates table that counts inactive days for each user in the end of the month. 


# In[33]:


no_act_e = no_act_e.groupby(['Id_str'], as_index = True)['TotalSteps'].size()
no_act_e


# In[34]:


no_act_b.shape[0] #counts the number of users who had at least one day with no activity at the beginning of the month


# In[35]:


no_act_e.shape[0] # counts the number of users who had at least one day with no activity at the end of the month. 


# In[38]:


no_act_s =ta[ta['TotalSteps']==0]
no_act_s
#creates a table with users who tracked sleep info and finds how many days they had with no activity. 


# #### Table shows that users who tracked sleep information had no days with zero activity. Users who logged sleep information also used their unit for other activities.

# In[34]:


no_act_w =wa[wa['TotalSteps']==0]
no_act_w
#creates a table with users who tracked weight info and finds how many days they had with no activity. 


# In[38]:


weight_count =wa.groupby('Id_str').count().reset_index()
weight_count


# #### Tables above show that users who tracked weight information had no days with zero activity. Users who logged weight information also used their unit for other activities.

# In[48]:


no_act = d[d['TotalSteps']==0] #creates a table for the entire month to count inactive days for users
no_act.head()


# In[49]:


no_act = no_act.groupby(['Id_str'], as_index = True)['TotalSteps'].size() 
# Groups inactive days by user and counts how many days of inactivity. 
no_act


# In[107]:


no_act.shape[0] # counts the total number of users who had at least one day of inactivity over the month. 


# #### Less than half the users have at least one day without the recommended daily activity, and 5 users have more than 9 days where they didn't track any activity.  

# In[112]:


plt.figure(figsize = (25, 13))
c= sns.lineplot(data=ta, x = 'ActiveTime' , y = 'Percent_sleep', ci = False ,hue='Id_str')
c.set_title("Percent Sleep vs Active Time.", fontsize=30);
plt.show


# #### Percent sleep is meant to be an indicator of quality of sleep. We see no correlation or trends from the graph between active time and how much sleep users got at the end of the day. 

# ##### Observations on use
# * Almost half the users missed at least one day of activity tracking.
# * There were more inactve days later in the month for the entire group.
# * Users who track more activities i.e. sleep and weight are more likely to use their device every day.
# * 4 users had 10 or more missed days of activity tracking. 
# * 5 users had no days where they got at least 30 min of recommended daily activity. 

# ## Act

# This analysis and the recommendations from the data are limited. We had a small sample set and a small window, so we are only able to do a descriptive analysis rather than trying to create a model for prediction. 
# 1. What are some trends in smart device usage?
#  * We can see that not all users are dedicated users. Further analysis using the hourly data can help us categorize users and create notifications to target users when they are most active with the device. 
#  * The fact that utilizing the weight and sleep function made for a more dedicated user leads us to think that ensuring users are aware and encouraged to use more of the features will create a more loyal user. 
#  * We can see that users used the device as positive feedback and that those that used the device more often had higher activitiy time, calories burned, and more steps.
# 2. How could these trends apply to Bellabeat customers?
#  * Time is a device similar to a fitbit so we can predict that users will wear the device almost daily and use it to track activity. Tailoring Time to Bellabeat customers using this information would be ideal. From the data we see that the more features the cutomers engage with on the product the more consistently they use the product. 
# 3. How could these trends help influence Bellabeat marketing strategy?
#  * Creating targeted notifications to encourage keeping up with good habits as well as notifying of trends in activity will help create more loyal users. 
#  * Creating a survey for customers when we expect the novelty of the wear off can help us tailor the experience for the users and encourage them to continue engagement with their product or upsell them to a subscription to add on more features. 
