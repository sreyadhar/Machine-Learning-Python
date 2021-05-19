
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # COST
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
from utils import *
import numpy as np

def compare_probs(prob1, prob2, epsilon):    
    return abs(prob1 - prob2) <= epsilon

def enforce_demographic_parity(categorical_results, epsilon):
         
    demographic_parity_data = {}
    thresholds = {}

    thresholds_final = []
    max_cost = np.NINF
    ppv_check = 0
    
    for ppv_check in np.arange(0,1,0.01):
        temp_thresholds = []
        for group in categorical_results.keys():
            threshold = 0
            while threshold<=1:
                thresholded_output = apply_threshold(categorical_results[group], threshold)
                pred_pos = get_num_predicted_positives(thresholded_output)/len(thresholded_output)
                if compare_probs(pred_pos,ppv_check,epsilon):
                    temp_thresholds.append(threshold)
                    break
                threshold+=0.01
        
        if len(temp_thresholds)==len(categorical_results.keys()):
            d = {}
            for index, group in enumerate(categorical_results.keys()):
                d[group] = apply_threshold(categorical_results[group], temp_thresholds[index])
            temp_cost = apply_financials(d)
            if temp_cost>max_cost:
                max_cost = temp_cost
                thresholds_final = temp_thresholds
        
    
    for index, group in enumerate(categorical_results.keys()):
        thresholds[group] = thresholds_final[index]
        demographic_parity_data[group] = apply_threshold(categorical_results[group], thresholds_final[index])       
    return demographic_parity_data, thresholds



#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}

    thresholds_final = []
    max_cost = np.NINF
    tpr_check = 0
    for tpr_check in np.arange(0,1,0.01):
        temp_thresholds = []
        for group in categorical_results.keys():
            threshold = 0
            for threshold in np.arange(0,1,0.01):
                thresholded_output = apply_threshold(categorical_results[group], threshold)
                curr_tpr = get_true_positive_rate(thresholded_output)
                if compare_probs(curr_tpr,tpr_check,epsilon):
                    temp_thresholds.append(threshold)
                    #break
        
        if len(temp_thresholds)==len(categorical_results.keys()):
            d = {}
            for index, group in enumerate(categorical_results.keys()):
                d[group] = apply_threshold(categorical_results[group], temp_thresholds[index])
            temp_cost = apply_financials(d)
            if temp_cost>max_cost:
                max_cost = temp_cost
                thresholds_final = temp_thresholds
       

    for index, group in enumerate(categorical_results.keys()):
        thresholds[group] =thresholds_final[index]
        equal_opportunity_data[group] = apply_threshold(categorical_results[group], thresholds_final[index])
    
    return equal_opportunity_data,thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    
    mp_data = {}
    thresholds = {}
    
    african_values=[]
    caucasian_values=[]
    hispanic_values=[]
    other_values=[]
    
    threshold = 0
    
    aa_data={}
    cc_data={}
    hp_data={}
    other_data={}
   
    for threshold in np.arange(0, 1, 0.01):
        african_american = apply_threshold(categorical_results['African-American'], threshold)
        caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        other = apply_threshold(categorical_results['Other'], threshold)
        
        aa_data['African-American'] = african_american
        cc_data['Caucasian'] = caucasian
        hp_data['Hispanic'] = hispanic
        other_data['Other'] = other
        
        african_values.append(apply_financials(aa_data))
        caucasian_values.append(apply_financials(cc_data))
        hispanic_values.append(apply_financials(hp_data))
        other_values.append(apply_financials(other_data))
        
   
    maximum_african = max(african_values)
    maximum_index_african = african_values.index(maximum_african)
    threshold_value_african = (maximum_index_african +1)/100
    
    maximum_cau = max(caucasian_values)
    maximum_index_cau = caucasian_values.index(maximum_cau)
    threshold_value_cau = (maximum_index_cau+1)/100
    
    maximum_his = max(hispanic_values)
    maximum_index_his = hispanic_values.index(maximum_his)
    threshold_value_his = (maximum_index_his+1)/100

    maximum_other = max(other_values)
    maximum_index_other = other_values.index(maximum_other)
    threshold_value_other = (maximum_index_other+1)/100
    
    african_american = apply_threshold(categorical_results['African-American'], threshold_value_african)
    caucasian = apply_threshold(categorical_results['Caucasian'], threshold_value_cau)
    hispanic = apply_threshold(categorical_results['Hispanic'], threshold_value_his)
    other = apply_threshold(categorical_results['Other'], threshold_value_other)
    
    thresholds['African-American'] = threshold_value_african
    thresholds['Caucasian'] = threshold_value_cau
    thresholds['Hispanic'] = threshold_value_his
    thresholds['Other'] = threshold_value_other
    
    mp_data['African-American'] = african_american
    mp_data['Caucasian'] = caucasian
    mp_data['Hispanic'] = hispanic
    mp_data['Other'] = other
    
    return mp_data, thresholds

    

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    thresholds_final = []
    max_cost = np.NINF
    ppvalue_check = 0
    
    for ppvalue_check in np.arange(0,1,0.01):
        temp_thresholds = []
        for group in categorical_results.keys():
            threshold = 0
            while threshold<=1:
                thresholded_output = apply_threshold(categorical_results[group], threshold)
                curr_ppv = get_positive_predictive_value(thresholded_output)
                if compare_probs(curr_ppv,ppvalue_check,epsilon):
                    temp_thresholds.append(threshold)
                    break
                threshold+=0.01

        if len(temp_thresholds)==len(categorical_results.keys()):
            d = {}
            for index, group in enumerate(categorical_results.keys()):
                d[group] = apply_threshold(categorical_results[group], temp_thresholds[index])
            temp_cost = apply_financials(d)
            if temp_cost > max_cost:
                max_cost = temp_cost
                thresholds_final = temp_thresholds
       

    for index, group in enumerate(categorical_results.keys()):
        thresholds[group] = thresholds_final[index]
        predictive_parity_data[group] = apply_threshold(categorical_results[group], thresholds_final[index])
    
    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    # Must complete this function!
    
    profit_values = []
    threshold = 0.0
    
    for threshold in np.arange (0, 1, 0.01):
        
        african_american = apply_threshold(categorical_results['African-American'], threshold)
        caucasian = apply_threshold(categorical_results['Caucasian'], threshold)
        hispanic = apply_threshold(categorical_results['Hispanic'], threshold)
        other = apply_threshold(categorical_results['Other'], threshold)
        
        single_threshold_data['African-American'] = african_american
        single_threshold_data['Caucasian'] = caucasian
        single_threshold_data['Hispanic'] = hispanic
        single_threshold_data['Other'] = other
        
        profit_values.append(apply_financials(single_threshold_data))
    
    max_profit = max(profit_values)
    threshold_values = []
    for value in range(len(profit_values)):
        if profit_values[value] == max_profit:
            threshold_values = (value + 1)/100
            
    african_american = apply_threshold(categorical_results['African-American'], threshold_values)
    caucasian = apply_threshold(categorical_results['Caucasian'], threshold_values)
    hispanic = apply_threshold(categorical_results['Hispanic'], threshold_values)
    other = apply_threshold(categorical_results['Other'], threshold_values)

    thresholds['African-American'] = threshold_values
    thresholds['Caucasian'] = threshold_values
    thresholds['Hispanic'] = threshold_values
    thresholds['Other'] = threshold_values

    single_threshold_data['African-American'] = african_american
    single_threshold_data['Caucasian'] = caucasian
    single_threshold_data['Hispanic'] = hispanic
    single_threshold_data['Other'] = other
     
    return single_threshold_data, thresholds
