import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score

seed = 67

df = pd.read_csv("churn-data.csv")

X, y = df.iloc[:, :-1], df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=seed)

def train_and_evaluate_model(model_constructor, criterion, max_depth):
    model = model_constructor(criterion=criterion, max_depth=max_depth)
    k_fold = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='f1')
    return model, np.mean(scores) 

def train_and_evaluate_models(model_constructor, depth_limits): 
    criterions = ["entropy", "gini"]
    models = []
    for depth_limit in depth_limits:
        for criterion in criterions:
            model, score = train_and_evaluate_model(model_constructor, criterion, depth_limit)
            models.append( {
                            "criterion": criterion,
                            "depth": depth_limit,
                            "model": model,
                            "score": score
                            } ) 
    return models

def select_the_best_model(models):
    best_score = 0
    best_model = {} 
    for model in models:
        score = model["score"]
        if score > best_score:
            best_score = score
            best_model = model
    print(f"The best model had criterion={best_model['criterion']}, max_depth={best_model['depth']}, and mean cross-validated F1 score={best_model['score']}")
    return best_model["model"] 
 
