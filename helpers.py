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