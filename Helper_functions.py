#!/usr/bin/env python
# coding: utf-8

# # Helper functions:  Mod4 project

# Be sure to import the required libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from collections import OrderedDict
from collections import defaultdict


# ### Get datetimes

# In[2]:


def get_datetimes(df):
    return pd.to_datetime(df.columns.values[7:], format='%Y-%m', errors = 'raise')


# ### Melt dataframe

# In[3]:


def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionID', 'Zip', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=False)
    melted = melted.dropna(subset=['value'])
    return melted 


# ### Function for creating a dataframe based on particular geographic unit (e.g., MetroState area, City, CountyName)

# In[4]:


# need to set df equal to dataframe with appropriate geographic grouping

def df_geog(df, col, geog):
    
    '''  From Helper_functions python file.
    Creates subset dataframe containing just the geographic unit 
    (e.g., 'MetroState' == 'Sacramento CA', 'City' == 'Davis', etc.) of interest.  
    It is necessary to set df equal to a dataframe with the appropriate geographic grouping: 
    e.g., to plot values by city in a metro area, df = df_metro_cities, col = 'MetroState',
    geog = 'Sacramento CA' (or metro area of interest). 
    '''
    df_metro_cities_geog = df.loc[df[col] == geog]
    return df_metro_cities_geog


# ### Function to print first n number of items in a dictionary
# 

# In[5]:


def print_first_n(dictionary, n):
    return {k: dictionary[k] for k in list(dictionary)[:n]}


# ### dict_cities_one_metro:  Function to create dictionary of *cities* in one *metro area*

# In[6]:


def dict_cities_one_metro(dict_metro_cities, metarea):
    dict_cities_one_metro = {metarea: dict_metro_cities[metarea]}
    return dict_cities_one_metro


# ### city_metro_list function creating *dataframe* AND *list* of cities in a metro area, *sorted from largest to smallest*

# In[8]:


# def city_metro_list(df_metro_cities, dict_sorted_metros_cities, col, metarea):
    
#     ''' From Helper_functions python file.
#     Function creates subset dataframe AND a list of cities in a metro area, sorted from largest to smallest by number of 
#     zip codes within that city'''
    
#     # Create subset dataframe containing just MetroState area in question
#     df_metro_cities_metarea = df_metro_cities.loc[df_metro_cities[col] == metarea]
#     ts = df_metro_cities_metarea
    
#     # Using dict_sorted_metros_cities, create list of all cities in the MetroState area of interest
#     metro_cities_metarea = dict_sorted_metros_cities[metarea]
    
#     # Create dictionary of cities and zips in metro area
#     from collections import defaultdict
#     dict_metarea_zips = defaultdict(list)
#     for idx,row in df_metro_zips_yr1.loc[df_metro_zips_yr1[col] == metarea].iterrows():
#         dict_metarea_zips[row['City']].append(row['Zip'])
        
#     # create dictionary containing the number of zip codes in each city
#     dict_num_metarea_zips = {key: len(dict_metarea_zips[key]) for key in dict_metarea_zips.keys()}
    
#     # Create Ordered dictionary sorting cities from most zip codes to least
#     ordict_metarea_zips_sorted = OrderedDict(sorted(dict_num_metarea_zips.items(), key=lambda item: item[1], reverse=True))
    
#     # create list of cities that is sorted by size (as determined by number of zip codes)
#     metarea_cities_sorted = list(ordict_metarea_zips_sorted.keys())
    
#     return ts, metarea_cities_sorted


# ### Plotting function:  plot_ts (plots zip codes on one axes in one figure)

# In[30]:


# # Framework from James Irving's study group for creating a plotting function that allows
# # entry of zip codes of interest

# def plot_ts(df_melt, col='value', zipcodes=['95616'], figsize=(12,8)):
    
#     ''' From Helper_functions python file.
#     Plots multiple zip code in a single axes/figure.  For each zip code, marks dates of:
#     1) maximum value reached during the housing bubble; 2) minimum value after the crash;
#     3) absolute minimum across entire time horizon (e.g., 1996 if you go back that far);
#     4) absolute minimum across entire time horizon (may or may not be the height of the bubble);
#     5) date when the national housing index (Case-Schiller) dropped.  
#     '''
    
#     fig, ax = plt.subplots()
    
#     for zc in zipcodes:
#         ts = df_melt[col].loc[df_melt['Zip'] == zc]
#         ts.plot(figsize=figsize, label = str(zc), ax=ax)

#     max_ = ts.loc['2004':'2010'].idxmax()  # 625600
#     crash = '01-2009'
#     min_ = ts.loc[crash:].idxmin()
#     max_all = ts.idxmax()
#     min_all = ts.idxmin()
#     mean_all = ts.mean()

#     ax.axvline('2004-01-01', label=f'2004', color = 'black')
#     ax.axvline('2011-12-01', label=f'2010', color = 'black')
#     ax.axvline(max_, label='Max Price in 2004 - 2010 timeframe', color = 'green', ls=':')
#     ax.axvline(crash, label = 'Housing Index Drops', color='black', ls=':')
#     ax.axvline(min_, label=f'Min Price Post Crash (2004 - 2010 timeframe) {min_}', color = 'red', ls=':')
#     ax.axvline(max_all, label='All time series max', color = 'green', ls=':')
#     ax.axvline(mean_all, label = 'All time series mean', color='red', ls=':')
#     ax.axvline(min_all, label=f'All time series min: {min_}', color = 'red', ls=':')

#     ax.legend()
    
#     return fig, ax


# ### Plotting function:  plot_single_geog (plots a single geographic unit)

# In[31]:


def plot_single_geog(df, col = 'value', col2 = 'MetroState', metunit = 'Sacramento CA', figsize=(12, 6)):
    
    ''' From Helper_functions python file.
    Plots housing values for individual geographic unit, e.g., MetroState, City, County.  
    Be sure to use the appropriate dataframe for the selected grouping (df_metro_cities for 
    cities in a metro area, for example).  Specify nrows, ncols, and figsize to match size of list.
    '''
    
    ts = df[col].loc[df[col2] == metunit]
    ax = ts.plot(figsize=figsize, title = metunit, label = 'Raw Price')

    max_ = ts.loc['2004':'2010'].idxmax()  
    crash = '01-2009'
    min_ = ts.loc[crash:].idxmin()
    val_2003 = ts.loc['2003-01-01']

    ax.axvline(max_, label='Max Price', color = 'green', ls=':')
    ax.axvline(crash, label = 'Housing Index Drops', color='red', ls=':')
    ax.axvline(min_, label=f'Min Price Post Crash {min_}', color = 'black')
    ax.axhline(val_2003, label='2003-01-01 value', color = 'blue', ls='-.', alpha=0.15)

    ax.legend()

# plot_ts_metros(df_metro_values, metros, col='value', x = (12, 8), nrows = 1, ncols = 1, legend=True, set_ylim = False, ylim = 1500000)
    
# plot_ts(df_melt, col='value', zipcodes=['95616'])


# ### Plotting function:  plot_ts_metros (plots values by metro area)

# In[10]:


# Adapted from James Irving's study group:   
    
def plot_ts_metros(df, metros, col='value', figsize = (18, 80), nrows = 15, ncols = 2, legend=True, set_ylim = True, ylim = 1500000):

    ''' From Helper_functions python file.
    Plots housing values by METRO area. Use dataframe that groups housing values at the 
    METRO (MetroState) level.  Need a list containing *subset* of METRO areas of interest.  
    *DON'T* run this for all metro areas; select a *subset* and create the appropriate
    list (e.g., top 30 metro areas).  Specify nrows, ncols, and figsize to match size of list.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, met in enumerate(metros, start=1):
        ax = fig.add_subplot(15,2,i)
        
        ts = df[col].loc[df['MetroState'] == met]
        ts.plot(title = str(met), ax=ax)

        max_ = ts.loc['2004':'2011'].idxmax()  
        crash = '01-2009'
        min_ = ts.loc[crash:].idxmin()
        val_2003 = ts.loc['2003-01-01']

        max_all = ts.idxmax()
        min_all = ts.idxmin()

        ax.axvline(max_, label=f'Max Price: {max_}', color = 'orange', ls=':')
        ax.axvline(crash, label = 'Housing Index Drops', color='black')
        ax.axvline(min_, label=f'Min Price Post Crash: {min_}', color = 'red', ls=':')
        ax.axvline(max_all, label=f'All time series max {max_all}', color = 'green', ls=':')
        ax.axvline(min_all, label=f'All time series min: {min_all}', color = 'red', ls=':')
        try:
            ax.axhline(val_2003, label='2003-01-01 value', color = 'blue', ls='-.', alpha=0.15)
        except:
            continue

        if set_ylim:
            ax.set_ylim(top=1000000)
        if legend:
            ax.legend()
    
        fig.tight_layout()
    
    return fig, ax


# ### Plotting function:   plot_ts_cities (plot values by city in a metro area)

# In[11]:


# Adapted from James Irving's study group:
    
def plot_ts_cities(df, cities, col='value', figsize = (18, 100), nrows=30, ncols=2, 
                   legend=True, set_ylim = False, ylim = 1400000):
    
    ''' From Helper_functions python file.
    Plots housing values by city within a metro area.  Need to use dataframe 
    with values by CITY for just that METRO (specify .loc in arguments that 
    column 'MetroState' == METRO). Need LIST of CITIES within that METRO area.  
    Specify nrows, ncols, and figsize to match size of dataset.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, city in enumerate(cities, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        
        ts = df[col].loc[df['City'] == city]
        ts.plot(title = str(city), ax=ax)

        max_ = ts.loc['2004':'2011'].idxmax()  
        crash = '01-2009'
        min_ = ts.loc[crash:].idxmin()
        val_2003 = ts.loc['2003-01-01']

        max_all = ts.idxmax()
        min_all = ts.idxmin()

        ax.axvline(max_, label=f'Max Price: {max_}', color = 'orange', ls=':')
        ax.axvline(crash, label = 'Housing Index Drops', color='black')
        ax.axvline(min_, label=f'Min Price Post Crash: {min_}', color = 'red', ls=':')
        ax.axvline(max_all, label=f'All time series max {max_all}', color = 'green', ls=':')
        ax.axvline(min_all, label=f'All time series min: {min_all}', color = 'red', ls=':')
        try:
            ax.axhline(val_2003, label='2003-01-01 value', color = 'blue', ls='-.', alpha=0.15)
        except:
            continue
            
        if set_ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(loc="best")

        fig.tight_layout()
    
    return fig, ax


# ### Plotting function:  plot_ts_metro_cities (plots of each city in each metro area)

# In[12]:


def plot_ts_metro_cities(df, dict_metro_cities, col='value', figsize = (18, 80), 
                         nrows=20, ncols=2, legend=False, set_ylim=False, ylim = 1400000):
    
    ''' From Helper_functions python file.
    Plots housing values of each CITY in each METRO area.  Need to use dataframe 
    grouping values by CITY.  Need DICTIONARY containing *subset* of CITIES by METRO areas 
    of choice.  *DON'T* run this for all metro areas; select a subset and create the appropriate
    dictionary (e.g., top 30 metro areas).  Specify nrows, ncols, and figsize to match size of dataset. 
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, key in enumerate(sorted(dict_metro_cities.keys()), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        for val in dict_metro_cities[key]:
            ts = df[col].loc[df['City'] == val]
            try: 
                max_ = ts.loc['2004':'2011'].idxmax()  
            except:
                continue

        ts.plot(title = f'{key}: {val}', ax=ax)

        max_ = ts.loc['2004':'2011'].idxmax()  
        crash = '01-2009'
        min_ = ts.loc[crash:].idxmin()
        val_2003 = ts.loc['2003-01-01']

        max_all = ts.idxmax()
        min_all = ts.idxmin()

        ax.axvline(max_, label=f'Max Price: {max_}', color = 'orange', ls=':')
        ax.axvline(crash, label = 'Housing Index Drops', color='black')
        ax.axvline(min_, label=f'Min Price Post Crash: {min_}', color = 'red', ls=':')
        ax.axvline(max_all, label=f'All time series max {max_all}', color = 'green', ls=':')
        ax.axvline(min_all, label=f'All time series min: {min_all}', color = 'red', ls=':')
        try:
            ax.axhline(val_2003, label='2003-01-01 value', color = 'blue', ls='-.', alpha=0.15)
        except:
            continue
        if set_ylim:
            ax.set_ylim(top=ylim)
        if legend:
            ax.legend()
    
        fig.tight_layout()
    
    return fig, ax


# ### Plotting function:  plot_ts_zips (plots individual zip codes in a list provided to the function)

# In[15]:


# Adapted from James Irving's study group:
    
def plot_ts_zips(df, zipcodes, col='value', figsize = (18, 60), nrows=15, ncols=2, 
                 legend=True, set_ylim = False, ylim = 800000):
    
    ''' From Helper_functions python file.
    Plots multiple zip codes in a single axes/figure.  For each zip code, marks dates of:
    1) maximum value reached during the housing bubble; 2) minimum value after the crash;
    3) absolute minimum across entire time horizon (e.g., 1996 if you go back that far);
    4) absolute minimum across entire time horizon (may or may not be the height of the bubble);
    5) date when the national housing index (Case-Schiller) dropped.  
    ''' 
    
    fig = plt.figure(figsize=figsize)
    
    for i, zc in enumerate(zipcodes, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        
        ts = df[col].loc[df['Zip'] == zc]
        ts.plot(title = zc, ax=ax)
        
        try: 
            max_ = ts.loc['2004':'2011'].idxmax()  
        except:
            continue

        crash = '01-2009'
        min_ = ts.loc[crash:].idxmin()
        val_2003 = ts.loc['2003-01-01']

        max_all = ts.idxmax()
        min_all = ts.idxmin()

        ax.axvline(max_, label=f'Max Price: {max_}', color = 'orange', ls=':')
        ax.axvline(crash, label = 'Housing Index Drops', color='black')
        ax.axvline(min_, label=f'Min Price Post Crash: {min_}', color = 'red', ls=':')
        ax.axvline(max_all, label=f'All time series max {max_all}', color = 'green', ls=':')
        ax.axvline(min_all, label=f'All time series min: {min_all}', color = 'red', ls=':')
        ax.axhline(val_2003, label='2003-01-01 value', color = 'blue', ls='-.', alpha=0.15)
        if set_ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(loc='best')
    
        fig.tight_layout()
    
    return fig, ax


# ### Plotting function:  plot_ts_zips_by_city (plot of each city with values by zip codes in a metro area)

# In[16]:


# Function below adapted from James Irving's study group:
    
def plot_ts_zips_by_city(df, dict_zips_cities,  nrows, ncols, col='value', 
                         figsize = (18, 100), legend=True, set_ylim = False, ylim = 1400000):
    
    '''  From Helper_functions python file.
    Plots zip codes by city within a metro area.  Need to use dataframe with values by 
    ZIP code and CITY for just that METRO (or specify .loc in arguments that 
    column 'MetroState' == METRO). Need DICTIONARY of ZIPS by CITY within that METRO area.  
    Specify nrows, ncols, and figsize to match size of dataset.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, key in enumerate(sorted(dict_zips_cities.keys()), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        for val in dict_zips_cities[key]:
            ts = df[col].loc[df['Zip'] == val]
            try: 
                max_ = ts.loc['2004':'2011'].idxmax()  
            except:
                continue
            
            crash = '01-2009'
#             min_val = ts.loc[crash:].min()   # purpose of var is to use to graph horizontal line at price at 2003-01-01
            min_ = ts.loc[crash:].idxmin()
            ts.plot(title = key, ax=ax)
#             val_2003 = ts.loc['2003-01-01']  # This is for graphic horizontal line at 2003 value, but can't 
                                               # get it to work on multiple zip plots 
            max_all = ts.idxmax()
            min_all = ts.idxmin()

            ax.axvline(max_, label = val, color = 'orange', ls=':')               
            ax.axvline(crash, color='black')                         # no labels
            ax.axvline(min_, color = 'red', ls=':')
            ax.axvline(max_all, color = 'green', ls=':')  
            ax.axvline(min_all, color = 'red', ls=':')  
            ax.axvline('2003-01-01', color = 'blue', ls='-.', alpha=0.15)   # for label, insert the following:  label='2003-01-01'
            
            if set_ylim:
                ax.set_ylim(ylim)
            if legend:
                ax.legend(loc="best")
    
        fig.tight_layout()
    
    return fig, ax

#  Ignore code below unless labels for each vertical line are desired

#  ax.axvline(max_, label=f'Max Price: {max_}', color = 'orange', ls=':')              # contains labels
#  ax.axvline(crash, label = 'Housing Index Drops', color='black')                     # contains labels
#  ax.axvline(min_, label=f'Min Price Post Crash: {min_}', color = 'red', ls=':')      # contains labels
#  ax.axhline(min_val, label=f'Min Price Post Crash: {min_}', color = 'red', ls='-.', alpha=0.15)  # can't get this to work on multiple zip plots 
#  ax.axvline(max_all, label=f'All time series max {max_all}', color = 'green', ls=':')  # contains labels
#  ax.axvline(min_all, label=f'All time series min: {min_all}', color = 'red', ls=':')   # contains labels
#  ax.axhline(ts.loc['2003-01-01'], label='2003-01-01 value', color = 'blue', ls='-.', alpha=0.15)  # can't get this to work on multiple zip plots 


# ### Plotting function:  city_zips_boxplot (boxplots of zip codes in a city)

# In[17]:


def city_zips_boxplot(df, city_zips, nrows, ncols, figsize=(18, 30)):
    
    ''' From Helper_functions python file.
    Plots boxplots of each zipcode within a city.  Need to use dataframe with values by ZIP 
    for just that CITY (or specify .loc in arguments that column 'City' == CITY in question).
    Need LIST of ZIP codes for that particular CITY.  Need to specify nrows, ncols, and figsize.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, zc in enumerate(city_zips, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ts = df.loc[df['Zip'] == zc]
        ts.boxplot(column = 'value', ax = ax)
        ax.set_title(f'{zc}')
        fig.tight_layout()


# ### Plotting function:  metro_zips_boxplot (boxplots by zip in metro area)

# In[18]:


def metro_zips_boxplot(df, metro_zips, nrows, ncols, figsize=(18, 100)):
    
    ''' From Helper_functions python file.
    Plots boxplots of each zipcode within a city.  Need to use dataframe with values by ZIP 
    for just that METRO area (or specify .loc in arguments that column 'MetroState' == METRO).
    Need a LIST of ZIP codes within that METRO area.  Need to specify nrows, ncols, and figsize.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, zc in enumerate(metro_zips, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ts = df.loc[df['Zip'] == zc]
        ts.boxplot(column = 'value', ax = ax)
        ax.set_title(f'{zc}')
        fig.tight_layout()


# ### Plotting function:  metro_cities_boxplot (boxplots by city in metro area)

# In[19]:


def metro_cities_boxplot(df, metro_cities, nrows, ncols, figsize=(18, 30)):
    
    ''' From Helper_functions python file.
    Plots boxplots of each city within a metro area.  Need to use dataframe with values 
    by CITY for just that METRO (or specify .loc in arguments that column 'MetroState' == METRO.
    Need LIST of CITIES within a particular metro area.  Need to specify nrows, ncols, and figsize.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, city in enumerate(metro_cities, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ts = df.loc[df['City'] == city]
        ts.boxplot(column = 'value', ax = ax)
        ax.set_title(f'{city}')
        fig.tight_layout()


# ### Plotting function:  metro_cities_zips_boxplot (boxplots by city and zip code in metro area)

# In[20]:


def metro_cities_zips_boxplot(df, dict_metro_cities_zips, nrows, ncols, figsize=(18, 40)):
    
    ''' From Helper_functions python file.
    Plots boxplots of all zip codes in each city within a metro area.  Need to use dataframe 
    with values by ZIP code and CITY for just that METRO (or specify .loc in arguments that 
    column 'MetroState' == METRO). Need DICTIONARY of ZIPS by CITY within that METRO area.  
    Need to specify nrows, ncols, and figsize.
    '''
    
    fig = plt.figure(figsize=figsize)
    
    for i, (city, zc) in enumerate(dict_metro_cities_zips.copy().items(), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ax.set_title(f'{city}')
        for i, zc in enumerate(dict_metro_cities_zips.copy()[city], start=1):
            ts = df.loc[df['Zip'] == zc]
            ts.boxplot(column = 'value', ax = ax)
            ax.set_title(f'{city}')
            fig.tight_layout()


# ### Creating ACF and PACF plots

# In[21]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot, lag_plot


# In[22]:


def plot_acf_pacf(ts, figsize=(10,6), lags=15):
    
    ''' From Helper_functions python file.
    Plots both ACF and PACF for given times series (ts).  Time series needs to be a Series 
    (not DataFrame) of values.  Can modify figsize and number of lags if desired.'''
    
    fig, ax = plt.subplots(nrows=2, figsize=figsize)
    plot_acf(ts, ax=ax[0], lags=lags)
    plot_pacf(ts, ax=ax[1], lags=lags)
    plt.tight_layout()
    
    for a in ax:
        a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=lags, integer=True))
        a.xaxis.grid()
    


# ### Seasonal Decomposition

# In[23]:


# from James Irving's study group
# plot seasonal decomposition

def plot_seasonal_decomp(ts):
    
    '''From Helper_functions python file'''

    decomp = seasonal_decompose(ts)
    ts_seasonal = decomp.seasonal

    ax = ts_seasonal.plot()
    fig = ax.get_figure()
    fig.set_size_inches(18,6)

    min_ = ts_seasonal.idxmin()
    max_ = ts_seasonal.idxmax()
    max_2 = ts_seasonal.loc[min_:].idxmax()
    min_2 = ts_seasonal.loc[max_2:].idxmin()


    ax.axvline(min_, label=min_, c='orange')
    ax.axvline(max_, c='orange', ls=':')
    ax.axvline(min_2, c='orange')
    ax.axvline(max_2, c='orange', ls=':')

    period = min_2 - min_ 
    ax.set_title(f'Season Length = {period}')
    
    return fig, ax


# ### Creating p, d, q, and m values for running ARIMA model

# In[24]:


import itertools

p_range = [0, 1, 2, 4, 8, 10]
q_range = range(0, 3)
d_range = range(1, 3)
m_range = (0, 6, 12)

pdq = list(itertools.product(p_range, d_range, q_range))
PDQM = list(itertools.product(p_range, d_range, q_range, m_range))


# In[25]:


def make_pdq_pdqm(p_range=(0,4), d_range=(0,3), q_range=(0,4), make_seasonal=True,
                  m_values=(0,12)):

    '''From Helper_functions python file'''
    
    import itertools
    p_values =range(p_range[0],p_range[1])
    d_values =range(d_range[0],d_range[1])
    q_values =range(q_range[0],q_range[1])
    
    params = {}
    params['pdq'] = list(itertools.product(p_values, d_values, q_values))
    
    if make_seasonal:
        params['PDQm'] = list(itertools.product(p_values, d_values, q_values, m_values))
    return params

# params = make_pdq_pdqm()


# ### Setting up functions for running ARIMA models

# In[26]:


import warnings
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# In[48]:


def arima_predict_error(X, arima_order):
    '''From Helper_functions python file'''
    train_size = int(len(X) * .85)
    train, test = X[0:train_size], X[train_size:]
    predictions = list()
    history = [x for x in train]
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp=0)
        y_hat = model_fit.forecast()[0]
        predictions.append(y_hat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error

def eval_arima_models(data, p_values, d_values, q_values):
    '''From Helper_functions python file'''
    data = data.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = arima_predict_error(data, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# In[49]:


# evaluate parameters
p_values = [0, 1, 2, 4, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
# eval_arima_models(ts.values, p_values, d_values, q_values)    


# ### Create ARIMA model and show summary results table

# In[50]:


# Define function

def arima_zipcode(ts, order):
    '''From Helper_functions python file'''
    model = ARIMA(ts.values, order = None)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    return model_fit


# ### Create forecast model

# In[51]:


# def forecast(model_fit, months=24, confint=2):
#     forecast = model_fit.forecast(months)
#     actual_forecast = forecast[0]
#     std_error = forecast[1]
#     forecast_confint = forecast[confint]
#     return actual_forecast, std_error, forecast_confint   


def forecast(model_fit, months=24, confint=2):
    '''From Helper_functions python file'''
    forecast = model_fit.forecast(months)
    actual_forecast = forecast[0]
    std_error = forecast[1]
    forecast_confint = forecast[confint]
    return actual_forecast, std_error, forecast_confint   


# ### Create dataframe to hold these values and join to existing dataframe

# In[52]:


def forecast_df(actual_forecast, std_error, forecast_confint, col = 'time', daterange = pd.date_range(start='2018-05-01', end='2020-04-01', freq='MS')):
    '''From Helper_functions python file'''
    df_forecast = pd.DataFrame({col: daterange})
    df_forecast['forecast'] = actual_forecast
    df_forecast['forecast_lower'] = forecast_confint[:, 0]
    df_forecast['forecast_upper'] = forecast_confint[:, 1]
    df_forecast.set_index('time', inplace=True)
    return df_forecast


# ### Create df_new with historical and forecasted values

# In[53]:


def concat_values_forecast(ts, df_forecast):
    '''From Helper_functions python file'''
    df_new = pd.concat([ts, df_forecast])
    df_new = df_new.rename(columns = {0: 'value'})
    return df_new


# ### Plot forecast results

# In[54]:


# Define function

def plot_forecast(df_new, figsize=(12,8), geog='95616'):
    '''From Helper_functions python file'''
    fig = plt.figure(figsize=figsize)
    plt.plot(df_new['value'], label='Raw Data')
    plt.plot(df_new['forecast'], label='Forecast')
    plt.fill_between(df_new.index, df_new['forecast_lower'], df_new['forecast_upper'], color='k', alpha = 0.2, 
                 label='Confidence Interval')
    plt.legend(loc = 'upper left')
    plt.title(f'Forecast for {geog}')


# ### Figure out percent change in home values

# In[55]:


# Define functions

def forecast_values(df_new, date = '2020-04-01'):
    '''From Helper_functions python file'''
    forecasted_price = df_new.loc[date, 'forecast']
    forecasted_lower = df_new.loc[date, 'forecast_lower']
    forecasted_upper = df_new.loc[date, 'forecast_upper']    
    return forecasted_price, forecasted_lower, forecasted_upper


# In[56]:


def last_value(df_new, date = '2018-04-01'):
    '''From Helper_functions python file'''
    last_value = df_new.loc[date, 'value']
    return last_value


# ### Compute and print predicted, best, and worst case scenarios

# In[1]:


# Define function

def pred_best_worst(pred, low, high, last, date='April 1, 2020'):
    '''From Helper_functions python file'''
    pred_pct_change = (((pred - last) / last) * 100)
    print(f'By the model prediction, I would expect to see a {round(pred_pct_change, 3)}% change in price by April 1, 2020.')
    lower_pct_change = ((low - last) / last) * 100
    print(f'At the lower bound of the confidence interval, I would expect to see a {round(lower_pct_change, 3)}% change in price by April 1, 2020.')
    upper_pct_change = ((high - last) / last) * 100
    print(f'At the upper bound of the confidence interval, I would expect to see a {round(upper_pct_change, 3)}% change in price by April 1, 2020.')
    return pred_pct_change, lower_pct_change, upper_pct_change


# In[ ]:




