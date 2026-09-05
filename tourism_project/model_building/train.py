
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load train/test data
# --------------------------------------------------

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

print("Training data:", Xtrain.shape)
print("Testing data:", Xtest.shape)

# --------------------------------------------------
# 2. Identify numerical and categorical columns
# --------------------------------------------------

categorical_columns = Xtrain.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = Xtrain.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# --------------------------------------------------
# 4. Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# --------------------------------------------------
# 5. Hyperparameter grid
# --------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

# --------------------------------------------------
# 6. MLflow experiment
# --------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Visit_With_Us_Tourism")

# --------------------------------------------------
# 7. Hyperparameter tuning
# --------------------------------------------------

with mlflow.start_run(run_name="RandomForest_GridSearch"):

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBest parameters:")
    print(best_params)

    # --------------------------------------------------
    # 8. Evaluate best model
    # --------------------------------------------------

    y_pred = best_model.predict(Xtest)
    y_prob = best_model.predict_proba(Xtest)[:, 1]

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred, zero_division=0)
    recall = recall_score(ytest, y_pred, zero_division=0)
    f1 = f1_score(ytest, y_pred, zero_division=0)
    roc_auc = roc_auc_score(ytest, y_prob)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(ytest, y_pred))

    print("\nClassification Report:")
    print(classification_report(ytest, y_pred, zero_division=0))

    # --------------------------------------------------
    # 9. Log parameters and metrics to MLflow
    # --------------------------------------------------

    clean_params = {
        key.replace("model__", ""): str(value)
        for key, value in best_params.items()
    }

    mlflow.log_params(clean_params)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    # Log the complete preprocessing + model pipeline
    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    # --------------------------------------------------
    # 10. Save best model for deployment
    # --------------------------------------------------

    os.makedirs("tourism_project/deployment", exist_ok=True)

    model_path = "tourism_project/deployment/best_model.pkl"

    joblib.dump(best_model, model_path)

    print("\nBest model saved to:")
    print(model_path)

print("\nModel training completed successfully!")
