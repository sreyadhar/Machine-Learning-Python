from sklearn.naive_bayes import MultinomialNB
from Preprocessing import preprocess
from Postprocessing import enforce_demographic_parity
from Report_Results import report_results
import numpy as np
from utils import *
epsilon=0.01
metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_demographic_parity(training_race_cases, epsilon)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

#print(test_race_cases)
output=[]
for key,value in test_race_cases.items():
    output.append(get_ROC_data(value,key))

plot_ROC_data(output)        
        
print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Accuracy on testing data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on testing data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")


print("Attempting to enforce demographic parity on test data...")

for group in test_race_cases.keys():
    num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
    prob = num_positive_predictions / len(test_race_cases[group])
    print("Probability of positive prediction for " + str(group) + ": " + str(prob))
        
print("")
for group in test_race_cases.keys():
    accuracy = get_num_correct(test_race_cases[group]) / len(test_race_cases[group])
    print("Accuracy for " + group + ": " + str(accuracy))

print("")
for group in test_race_cases.keys():
    FPR = get_false_positive_rate(test_race_cases[group])
    print("FPR for " + group + ": " + str(FPR))

print("")
for group in test_race_cases.keys():
    FNR = get_false_negative_rate(test_race_cases[group])
    print("FNR for " + group + ": " + str(FNR))

print("")
for group in test_race_cases.keys():
    TPR = get_true_positive_rate(test_race_cases[group])
    print("TPR for " + group + ": " + str(TPR))

print("")
for group in test_race_cases.keys():
    TNR = get_true_negative_rate(test_race_cases[group])
    print("TNR for " + group + ": " + str(TNR))


print("")
total_cost = apply_financials(test_race_cases)
print("Total cost: ")
print('${:,.0f}'.format(total_cost))
total_accuracy = get_total_accuracy(test_race_cases)
print("Total accuracy: " + str(total_accuracy))
print("-----------------------------------------------------------------")
print("")


