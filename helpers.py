import numpy as np
import itertools

def batch_processing(model=None, X_subset=None, Y_subset=None, n_batches = 101):
    """
    Takes in X and Y subsets and fits a model with it. Returns the model after processing all batches.
    """
    
    assert X_subset.shape[0] == Y_subset.shape[0]
    
    batch_size = X_subset.shape[0] // n_batches

    it = itertools.count(step=batch_size)

    for _ in range(n_batches):
        start = next(it)
        end = start + batch_size
        
        model.fit(X_subset[start : end], Y_subset[start : end])
        
        print(f"{end} trained examples. {round(end / X_subset.shape[0] * 100, 2)}%")

        if (end + batch_size) > X_subset.shape[0]:
            assert (end + len(X_subset[end:])) == X_subset.shape[0]
            model.fit(X_subset[end:], Y_subset[end:])
            print(f"{end + len(X_subset[end:])} trained examples. {round((end + len(X_subset[end:])) / X_subset.shape[0] * 100, 2)}%")
    
    return model

def report_classification(y_test, y_pred):
    """
    Takes in Y_test and Y_pred and returns precision, recall and F1 score 
    for every feature in the dataset, and the overall accuracy of the model.
    
    Input:
        Y_test (pandas.core.series.Series): a subset of Y with the purpose of testing the model
        Y_pred (pandas.core.series.Series): predictions made with X_test by the model
        
    Output:
        Prints out the following format
            feature_name
                Precision: __%
                Recall: __%
                F1 Score: __%
                
                ...
                
                Accuracy Score: __%
                
        And also returns the full value of accuracy.
    """
    
    for idx, col in enumerate(y_test):
        set_y_pair = (y_test[col], y_pred[:, idx])
        avg='weighted'
        rep_col = "{}\n\tPrecision: {:.2f}%\n\tRecall: {:.2f}%\n\tF1 Score: {:.2f}%\n".format(col,
                                                                                 precision_score(*set_y_pair, average=avg), 
                                                                                 recall_score(*set_y_pair, average=avg), 
                                                                                 f1_score(*set_y_pair, average=avg))
        print(rep_col)
        
    print('Accuracy Score: {:.2f}%'.format(np.mean(y_test.values == y_pred)))

    return np.mean(y_test.values == y_pred)