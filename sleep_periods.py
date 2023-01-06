import LAMP
import os
LAMP.connect(os.environ["LAMP_ACCESS_KEY"], os.environ["LAMP_SECRET_KEY"], os.environ["LAMP_SERVER_ADDRESS"])
import sys
import cortex
import scipy as sc
import matplotlib.pyplot as plt
import skimage.filters as filt
import math
import pandas as pd
import numpy as np
from statistics import mode
import scipy
import datetime
import time
import logging
import random as random
from sklearn.preprocessing import MinMaxScaler
import altair as alt
time_factor = 1
ss_limit = 1000000
ac_limit = 1000000
MS_PER_DAY=1000*60*60*24
MS_PER_HOUR=1000*60*60
import scipy.stats as stats
import random as random
import warnings

""" Module for computing screen active bouts from screen state """

from cortex.raw.accelerometer import accelerometer
from cortex.raw.screen_state import screen_state



# # Random walk method      !!!!
# Note: unlike other methods, this method must start near midday (automatic shift of start and end 
# to nearest hour, differences will cause error)
# hour must be specified in 24hr format.
# there are some days where the missingness is complete, as in no data at all. therefore, we need to add parameter for prior based on results for previous day,
# constantly updating(?)
# perhaps mean of last week? can add many variables and priors based on different categories

def time_shift(timestamp, hour):
    """Get the nearest timestamp at the hour specified
    Ex: For time = 5pm, 2am shifts to previous 5pm (9 hours apart)
    and 6am shifts to next 5pm (11 hours apart)
    Purpose: Need sleep periods to be fully contained within window
    """
    dt = datetime.datetime.fromtimestamp(int(timestamp/1000)).strftime('%Y-%m-%d')
    dt_splits = [int(x) for x in dt.split('-')]
    dt_shift_ts = datetime.datetime(dt_splits[0], dt_splits[1], dt_splits[2], hour, 0, 0).timestamp()*1000
    if (timestamp-dt_shift_ts) > 12*MS_PER_HOUR:
        dt_shift_ts = dt_shift_ts + 24*MS_PER_HOUR
    elif (dt_shift_ts-timestamp) > 12*MS_PER_HOUR:
        dt_shift_ts = dt_shift_ts - 24*MS_PER_HOUR
    return dt_shift_ts

def sleep_periods(part_id, target_acc = 250, **kwargs):

    test_id = part_id
    # get closest 12pm EST
    st = kwargs['start']
    nd = kwargs['end']
    start_shifted = time_shift(st, 17)
    end_shifted = time_shift(nd, 17)

    start = start_shifted
    end = end_shifted

    """print(f"Starting at: {datetime.datetime.fromtimestamp(int(start)/1000).strftime('%Y-%m-%d at %H:%S UTC')}, \
Ending at: {datetime.datetime.fromtimestamp(int(end/1000)).strftime('%Y-%m-%d at %H:%S UTC')}")"""
    
    float_days = (end - start)/MS_PER_DAY
    n_days = int(float_days)
    n_chunk = 6*n_days
    chunk_size = (end - start)/n_chunk

    dat_ss = []
    dat_ac = []
    for i in range(0, n_chunk):
        dat_piece_ss=screen_state(id = test_id, start=int(start+i*chunk_size), end=int(start+(i+1)*chunk_size), _limit=ss_limit)['data']
        dat_piece_ac=accelerometer(id = test_id, start=int(start+i*chunk_size), end=int(start+(i+1)*chunk_size), _limit=ac_limit)['data']
        dat_ss += dat_piece_ss
        dat_ac += dat_piece_ac
    ss_df = pd.DataFrame(dat_ss)
    acc_df = pd.DataFrame(dat_ac)

    if len(dat_ac) == 0 or len(dat_ss) == 0:
        print('No data in this window, returning None')
        return {"values": None, "chart": None}
    
    elif len(dat_ac) > 0:
        length = len(acc_df['timestamp'])
        keep = np.zeros(length)
        num = int(length/1)
        keep_inds = np.linspace(0, length-1, num).astype(int)
        for ind in keep_inds:
            keep[ind]=1
        acc_df['keep'] = keep
        acc_df=acc_df[acc_df['keep']==1].reset_index()
        acc_df = acc_df.sort_values(by='timestamp')
        for dim in ['x', 'y', 'z']:
            acc_df[f'delta_{dim}'] = acc_df[dim] - acc_df[dim].shift()
        acc_df['mag']=np.sqrt(acc_df['delta_x']**2+acc_df['delta_y']**2+acc_df['delta_z']**2) 

    on_events = [0, 3]
    off_events = [1, 2]

    sc_timestamps = [point['timestamp'] for point in dat_ss]
    sc_dat = pd.DataFrame(sc_timestamps, columns=['timestamp']).sort_values(by='timestamp')
    
    ss_df['sleep'] = np.where(ss_df['value'].isin(on_events), 1, 0)
    
    sc_dat = ss_df

    bin_size = (0.25)*MS_PER_HOUR
    n_bins = int((end-start) / bin_size)
    time_bins = []
    time_counts = []
    time_counts_ac = []
    time_dt = []
    for i in range(0, n_bins):
        rel_dat = sc_dat[sc_dat['timestamp'] >= start + i*bin_size]
        rel_dat = rel_dat[rel_dat['timestamp'] < start + (i+1)*bin_size]
        rel_dat_ac = acc_df[acc_df['timestamp'] >= (start + i*bin_size)]
        rel_dat_ac = rel_dat_ac[rel_dat_ac['timestamp'] < (start + (i+1)*bin_size)]
        count = np.sum(rel_dat['sleep'])
        if len(rel_dat_ac) > 0:
            count_ac = np.sum(rel_dat_ac['mag'])
        else:
            count_ac = None
        time_bins.append(start + i*bin_size)
        time_counts.append(count)
        time_counts_ac.append(count_ac)
        time_dt.append(datetime.datetime.fromtimestamp((start + i*bin_size)/1000))

    count_df = pd.DataFrame(time_bins, columns = ['bins'])
    count_df['hours'] = (count_df['bins'] - min(count_df['bins']))/MS_PER_HOUR
    count_df['counts'] = time_counts
    count_df['counts_ac'] = time_counts_ac
    count_df['dt'] = time_dt
    # normalize ac so that the sum of all data is equal to device state ...
    def rescale(dat, A):
        norm_sum = np.sum(dat) / A
        return norm_sum
    parameter, cov = sc.optimize.curve_fit(rescale, [np.sum(count_df['counts_ac'])], [np.sum(count_df['counts'])], maxfev=1000)
    count_df['ac_rescaled'] = count_df['counts_ac'] / parameter
    
    if abs(np.sum(count_df['ac_rescaled']) - np.sum(count_df['counts'])) > 1:
        print("Unable to rescale accelerometer data")
        return {"values": None, "chart": None}

    warnings.filterwarnings('ignore')
    # format of x: ts, ta, la_ss, la_ac
    transition_model = lambda x: [np.random.normal(loc=x[0],scale=0.2,size=1)[0],
                                  np.random.normal(loc=x[1],scale=0.2,size=1)[0],
                                  np.random.normal(loc=x[2],scale=0.1,size=1)[0],
                                  np.random.normal(loc=x[3],scale=0.1,size=1)[0]]

    def prior(x, start, end):
        if len([f for f in x if f < 0]) > 0:
            return 0
        elif (x[0] > x[1]) or (x[0] < start) or (x[1] > end):
            return 0
        elif (x[1] - x[0]) <= 2:
            return 2*scipy.stats.norm.cdf(x[1]-x[0], loc=2, scale=1)
        else:
            return 1

    def manual_log_like_normal(x, data):
        # data must be in form of DataFrame with columns:
        # hours, counts (screen state), and ac_rescaled (accelerometer)

        df = data[['hours', 'counts', 'ac_rescaled']]
        df.columns = ['time', 'counts', 'counts_ac']
        # format of x: (timestamps) ts, ta, la_ss, la_ac

        ts = x[0]
        ta = x[1]

        conditions_asleep = [(df['time'] >= ts) & (df['time'] < ta)]

        state = np.where(conditions_asleep, 1, 0)[0]
        df['state'] = state

        conditions_awake = [(df['state'] == 0) & (df['counts'] > x[2]),
                            (df['state'] == 0) & (df['counts'] <= x[2]),
                            (df['state'] == 1) & (df['counts'] > -1)]

        conditions_awake_ac = [(df['state'] == 0) & (df['counts_ac'] > x[3]),
                               (df['state'] == 0) & (df['counts_ac'] <= x[3]),
                               (df['state'] == 1) & (df['counts_ac'] > -1)]

        values_list = [1 - scipy.stats.poisson.cdf(df['counts'], x[2]),
                       scipy.stats.poisson.cdf(df['counts'], x[2]),
                       1 - scipy.stats.expon.cdf(df['counts'], loc=0, scale=0.25)]

        values_list_ac = [1 - scipy.stats.poisson.cdf(df['counts_ac'], x[3]),
                          scipy.stats.poisson.cdf(df['counts_ac'], x[3]),
                          1 - scipy.stats.expon.cdf(df['counts_ac'], loc=0, scale=0.25)]

        like_list = np.select(conditions_awake, values_list)
        like_list_ac = np.select(conditions_awake_ac, values_list_ac)

        likelihood = like_list[like_list > 0].prod() * like_list_ac[like_list_ac > 0].prod()

        #likelihood = df[df['likelihood'] > 0]['likelihood'].prod()

        return likelihood

    #acceptance criteria
    def acceptance(x, x_new):
        if x_new > x:
            return True
        else:
            accept = np.random.uniform(0, 1)
            if (accept < (x_new/x)):
                return True
            else:
                return False

    def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,data,acceptance_rule,target_acc):
        # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
        # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
        # param_init: a starting sample
        # iterations: number of accepted to generated
        # data: the data that we wish to model
        # acceptance_rule(x,x_new): decides whether to accept or reject the new sample
        x = param_init
        accepted = []
        rejected = []   
        likelihoods = []
        times = data['hours']
        start = min(times)
        end = max(times)
        stage = 0
        #for i in range(iterations):
        while len(accepted) < target_acc:
            if len(accepted) > stage*(target_acc/10):
                #print(round(len(accepted)/target_acc, 2))
                stage+=1
            x_new = transition_model(x)
            x_lik = likelihood_computer(x, data) * prior(x, start, end)
            x_new_lik = likelihood_computer(x_new, data) * prior(x_new, start, end)
            if(acceptance_rule(x_lik, x_new_lik)):
                x = x_new
                accepted.append(x_new)
                likelihoods.append(x_lik)
            else:
                rejected.append(x_new)

        return np.array(accepted), np.array(rejected), np.array(likelihoods)
    
    def double_gauss(bins, u, u2, dev, dev2, a, b):
    # bins is the x data (the bins)
        gauss1 = a*np.exp((-1/2)*(bins-u)**2/(dev**2))
        gauss2 = b*np.exp((-1/2)*(bins-u2)**2/(dev2**2))
        freq = gauss1 + gauss2
        return freq

    starting_time = cortex.now()
    start = min(count_df['hours'])
    end = max(count_df['hours'])
    n_days = int((end - start + 1)/24)
    onsets = []
    offsets = []
    guess_qual = []
    accepted_list = []
    day_starts = []
    fig1 = plt.figure(figsize = (3*n_days, 7))
    for x in range(0, n_days):
        start_day = start + x*24
        end_day = start_day + 24
        day_starts.append(start_day)
        input_df = count_df[count_df['hours'] >= start_day]
        input_df = input_df[input_df['hours'] < end_day]
        starting_hour = np.median(input_df[input_df['counts'] == 0]['hours'])
        target_acc = target_acc
        accepted, rejected, likelihoods = metropolis_hastings(manual_log_like_normal,
                                                 prior,
                                                 transition_model,
                                                 # format of x: ts, ta, mu_ac_sleep, la_ss, mu_ac, var_ac, var_ac_sleep
                                                 [starting_hour-1, starting_hour+1, 3, 3],
                                                 input_df,
                                                 acceptance,
                                                 target_acc = target_acc)
        accepted_list.append(accepted[-50:len(accepted)])
        ts_hist, ts_bin_edges = np.histogram([f[0] for f in accepted[-500:len(accepted)]], bins=np.arange(start_day, end_day, 1))
        ta_hist, ta_bin_edges = np.histogram([f[1] for f in accepted[-500:len(accepted)]], bins=np.arange(start_day, end_day, 1))
        plt.plot([f[0] for f in accepted])
        plt.plot([f[1] for f in accepted])
        #ts = (ts_bin_edges[np.argmax(ts_hist)] + ts_bin_edges[np.argmax(ts_hist) + 1]) / 2
        #ta = (ta_bin_edges[np.argmax(ta_hist)] + ta_bin_edges[np.argmax(ta_hist) + 1]) / 2
        ts = np.round(np.median([f[0] for f in accepted[-int(target_acc/2):]]), 2)
        ta = np.round(np.median([f[1] for f in accepted[-int(target_acc/2):]]), 2)
        ts_qual = len([f[0] for f in accepted[-int(target_acc/2):] if (f[0] < (ts + 0.25)) and (f[0] > (ts - 0.25))]) / int(target_acc/2)
        ta_qual = len([f[1] for f in accepted[-int(target_acc/2):] if (f[1] < (ta + 0.25)) and (f[1] > (ta - 0.25))]) / int(target_acc/2)
        confidence = np.round(100*(ts_qual + ta_qual)/2, 2)
        like = np.median(likelihoods)
        dur = ta - ts
        print(f"Onset: {ts}, Offset: {ta}. Duration: {dur} hours. Confidence: {confidence}%")
        onsets.append(ts)
        offsets.append(ta)
        guess_qual.append(confidence)
    fig2 = plt.figure(figsize = (3*n_days, 7))
    plt.ylabel('Relative Activity')
    plt.xlabel('Hours Elapsed')
    plt.plot(count_df['hours'], count_df['counts'], label = 'Device State Activity Count')
    plt.plot(count_df['hours'], count_df['ac_rescaled'], label = 'Accelerometer Magnitude Sum (Rescaled)')
    
    chart_ac = alt.Chart(count_df).mark_line(color='blue').encode(
        x = alt.X('hours', title='Hours Elapsed'),
        y = alt.Y('ac_rescaled', title='Relative Activity', scale=alt.Scale(domain=(0, 1.5*max(count_df['ac_rescaled'])))))
    chart_count = alt.Chart(count_df).mark_line(color='orange').encode(
        x = alt.X('hours', title='Hours Elapsed'),
        y = alt.Y('counts', title='Relative Activity', scale=alt.Scale(domain=(0, 1.5*max(count_df['ac_rescaled'])))))
    chart = chart_ac + chart_count
    
    for day in day_starts:
        plt.axvline(day, color='gray')
    for on in onsets:
        plt.axvline(on, color='black')
    for off in offsets:
        plt.axvline(off, color='black')
        
    starts_df = pd.DataFrame(day_starts, columns = ['Day Start'])
    
    rules_start = alt.Chart(starts_df).mark_rule(color='black').encode(
        x='Day Start',
        strokeWidth=alt.value(2)  
    )
    
    on_df = pd.DataFrame(onsets, columns = ['onsets'])
    off_df = pd.DataFrame(offsets, columns = ['offsets'])
    
    rules_on = alt.Chart(on_df).mark_rule(color='gray').encode(
        x='onsets',
        strokeWidth=alt.value(2) 
    )
    
    rules_off = alt.Chart(off_df).mark_rule(color='gray').encode(
        x='offsets',
        strokeWidth=alt.value(2) 
    )
        
    plt.show()

    ending_time = cortex.now()
    runtime = (ending_time - starting_time)/(1000*60)
    print(f"{runtime} mins")
    
    return {"values": [onsets, offsets, guess_qual, day_starts], "chart": chart + rules_start + rules_on + rules_off}
