from sklearn.tree import DecisionTreeClassifier

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    clf = DecisionTreeClassifier(min_samples_split=2)
    clf.fit(features_train, labels_train)

    
    
    return clf