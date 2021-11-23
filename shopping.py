import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # print('LOAD_DATA')
    evidence = []
    labels = []
    with open(filename) as csvfile:
        reader = list(csv.reader(csvfile))

        for i in reader[1:]:  
            evidence.append(numerify(i[:17])) 
            labels.append(boole_str_to_int(i[17]))

    return(evidence, labels)

def numerify(data):
    # print('NUMERIFY')
    for i in (0,2,4,11,12,13,14): #15 i 16 - boole
        data[i] = int(data[i])

    for i in (1,3,5,6,7,8,9):
        data[i] = float(data[i])
    
    data[10] = month_number(data[10])

    data[15] = is_returning(data[15])
    data[16] = boole_str_to_int(data[16])

    return data

def month_number(month):
    month = month[:3]
    months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

    return months.index(month)

def boole_str_to_int(s=str):
    if s == 'TRUE':
        return 1
    return 0

def is_returning(s):
    if s == 'Returning_Visitor':
        return 1
    return 0


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors=1)

    return model.fit(evidence, labels)

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # print('EVALUATE')
    # print('labels:', labels)
    # print('predictions:', predictions)
    
    true_positive_count = 0
    false_positive_count = 0
    true_negative_count = 0
    false_negative_count = 0

    for i, value in enumerate(predictions):
        if value == 1 and labels[i] == 1:
            true_positive_count += 1
        elif value == 1 and labels[i] == 0:
            false_positive_count += 1
        elif value == 0 and labels[i] == 0:
            true_negative_count += 1
        elif value == 0 and labels[i] == 1:
            false_negative_count += 1

    
    sensitivity = true_positive_count / (true_positive_count + false_negative_count)
    specificity = true_negative_count / (true_negative_count + false_positive_count)

    return(sensitivity, specificity)


if __name__ == "__main__":
    main()
