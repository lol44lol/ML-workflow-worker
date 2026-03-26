from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error
from importlib import import_module
import hashlib
import os
import pandas as pd
import joblib
import json
import metrics


from sklearn.pipeline import Pipeline

METRICS = metrics.METRICS
MODEL_REGISTRY = {

    # =========================
    # LINEAR MODELS
    # =========================
    "linear_regression": {
        "import": ("sklearn.linear_model", "LinearRegression"),
        "task": "regression",
        "metrics": ["mse", "rmse", "mae", "r2"]
    },
    "ridge": {
        "import": ("sklearn.linear_model", "Ridge"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },
    "lasso": {
        "import": ("sklearn.linear_model", "Lasso"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },
    "elastic_net": {
        "import": ("sklearn.linear_model", "ElasticNet"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },
    "logistic_regression": {
        "import": ("sklearn.linear_model", "LogisticRegression"),
        "task": "classification",
        "metrics": ["accuracy", "precision", "recall", "f1"]
    },

    # =========================
    # TREE MODELS
    # =========================
    "decision_tree_classifier": {
        "import": ("sklearn.tree", "DecisionTreeClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "decision_tree_regressor": {
        "import": ("sklearn.tree", "DecisionTreeRegressor"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },

    # =========================
    # ENSEMBLE METHODS
    # =========================
    "random_forest_classifier": {
        "import": ("sklearn.ensemble", "RandomForestClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "precision", "recall", "f1"]
    },
    "random_forest_regressor": {
        "import": ("sklearn.ensemble", "RandomForestRegressor"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },
    "extra_trees_classifier": {
        "import": ("sklearn.ensemble", "ExtraTreesClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "extra_trees_regressor": {
        "import": ("sklearn.ensemble", "ExtraTreesRegressor"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },
    "gradient_boosting_classifier": {
        "import": ("sklearn.ensemble", "GradientBoostingClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "gradient_boosting_regressor": {
        "import": ("sklearn.ensemble", "GradientBoostingRegressor"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },
    "adaboost_classifier": {
        "import": ("sklearn.ensemble", "AdaBoostClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },

    # =========================
    # MODERN BOOSTING
    # =========================
    "hist_gradient_boosting_classifier": {
        "import": ("sklearn.ensemble", "HistGradientBoostingClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "hist_gradient_boosting_regressor": {
        "import": ("sklearn.ensemble", "HistGradientBoostingRegressor"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },

    # =========================
    # SVM
    # =========================
    "svm_classifier": {
        "import": ("sklearn.svm", "SVC"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "svm_regressor": {
        "import": ("sklearn.svm", "SVR"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },

    # =========================
    # KNN
    # =========================
    "knn_classifier": {
        "import": ("sklearn.neighbors", "KNeighborsClassifier"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "knn_regressor": {
        "import": ("sklearn.neighbors", "KNeighborsRegressor"),
        "task": "regression",
        "metrics": ["mse", "r2"]
    },

    # =========================
    # NAIVE BAYES
    # =========================
    "gaussian_nb": {
        "import": ("sklearn.naive_bayes", "GaussianNB"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "multinomial_nb": {
        "import": ("sklearn.naive_bayes", "MultinomialNB"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },
    "bernoulli_nb": {
        "import": ("sklearn.naive_bayes", "BernoulliNB"),
        "task": "classification",
        "metrics": ["accuracy", "f1"]
    },

    # =========================
    # CLUSTERING (NO LABELS)
    # =========================
    "kmeans": {
        "import": ("sklearn.cluster", "KMeans"),
        "task": "clustering",
        "metrics": ["inertia", "silhouette"]
    },
    "dbscan": {
        "import": ("sklearn.cluster", "DBSCAN"),
        "task": "clustering",
        "metrics": ["silhouette"]
    },

    # =========================
    # DIMENSIONALITY REDUCTION
    # =========================
    "pca": {
        "import": ("sklearn.decomposition", "PCA"),
        "task": "unsupervised",
        "metrics": ["explained_variance_ratio"]
    },
    "svd": {
        "import": ("sklearn.decomposition", "TruncatedSVD"),
        "task": "unsupervised",
        "metrics": ["explained_variance_ratio"]
    }
}


class Models:
    def __init__(self):
        
        self.models = {

        }

    def load_data(self,file_name,pipeline_name):
        base_path = f"./transformed-data/{file_name}/{pipeline_name}"
        X_train = pd.read_parquet(f"{base_path}/X_train.parquet")
        X_test = pd.read_parquet(f"{base_path}/X_test.parquet")
        Y_train = pd.read_parquet(f"{base_path}/Y_train.parquet").squeeze()
        Y_test = pd.read_parquet(f"{base_path}/Y_test.parquet").squeeze()
        print(X_train)
        return (X_train,X_test,Y_train,Y_test)
    

    def _load_pipelines(self,file_name,pipeline_name):
        encoder_path =  f"encoders/{file_name}/{pipeline_name}.pk"
        scaler_path =  f"scalers/{file_name}/{pipeline_name}.pk"
        encoder = None
        scaler = None
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

        return (encoder,scaler)

    def save_model(self,model,file_name,pipeline_name,experiment_name,model_name,arguments):
        encoder,scaler = self._load_pipelines(file_name,pipeline_name)
        pipes = []
        if encoder:
            pipes.append(encoder)
        if scaler:
            pipes.append(scaler)
        pipes.append(model)

        pipeline = Pipeline(pipes)
        base_path = f"./models/{file_name}/{pipeline_name}"
        model_name = model_name + "_"+experiment_name + "_" + hashlib.sha256(json.dumps(arguments).encode()).hexdigest()
        os.makedirs(base_path,exist_ok=True)
        joblib.dump(pipeline,f"{base_path}/{model_name}.pk1")
    
    async def build_model(self,payload):
        selected_model = payload['model_name']
        experiment_name = payload['experiment_name']
        file_name = payload['file_name']
        pipeline_name = payload['pipeline_name']
        arguments  = payload['arguments']
        metrics  = payload['metrics'] or []
        self.models[experiment_name] = {
            "selected_model":selected_model,
            "file_name":file_name,
            "pipeline_name":pipeline_name
        }


        model  = self.get_model_import(selected_model,**arguments)
        X_train,X_test,Y_train,Y_test = self.load_data(file_name,pipeline_name)
        model.fit(X_train,Y_train)
        Y_predicted = model.predict(X_test)
        metrics_output = []
        for metric in metrics:
            metrics_output.append({metric: METRICS[metric](Y_test,Y_predicted)})


        self.save_model(model,file_name,pipeline_name,experiment_name,selected_model,arguments)

        return {"pipeline_name":pipeline_name,
                "file_name":file_name,
                "metrics":metrics_output,
                "experiment_name":experiment_name,
                "model_name":selected_model,
                "arguments":arguments
                }
    

    def get_model_import(self,model_name,**kwargs):
        module_path,class_name = MODEL_REGISTRY[model_name]['import']
        module = import_module(module_path)
        model_class = getattr(module,class_name)

        model = model_class(**kwargs)

        return model