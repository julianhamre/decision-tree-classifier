def permutation_importance(model, X, y, metric, n_repeats, seed):
    X = X.copy().reset_index(drop=True)
    reference_score = metric(y, model.predict(X))
    feature_importances = []
    for column_name in X.columns:
        permutation_scores = []
        original_column = X[column_name]
        for k in range(n_repeats):
            shuffled_column = shuffle_column(original_column, seed + k)
            X[column_name] = shuffled_column 
            permutation_scores.append(metric(y, model.predict(X)))
        feature_importances.append(feature_importance(reference_score, permutation_scores))
        X[column_name] = original_column
    return feature_importances

def shuffle_column(column, seed):
    return column.sample(frac=1, random_state=seed).values

def feature_importance(reference_score, permutation_scores):
    return reference_score - 1 / len(permutation_scores) * sum(permutation_scores)
