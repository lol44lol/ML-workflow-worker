from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error
from importlib import import_module
import hashlib
import os
import pandas as pd
import joblib
import json
import metrics
from pathlib import Path


from sklearn.pipeline import Pipeline

def transform_arguments(raw_args):
    """
    Generalized transformer for all Scikit-Learn models.
    Converts Frontend strings/nulls into Python types.
    """
    # 1. Define how each key name should be cast
    # This covers every key in your TypeScript MODEL_ARGUMENTS
    casting_rules = {
        # Integer Types
        "n_estimators": int,
        "max_iter": int,
        "max_depth": lambda x: int(x) if x is not None and str(x).lower() != 'null' else None,
        "min_samples_split": int,
        "min_samples_leaf": int,
        "n_neighbors": int,
        
        # Float Types
        "alpha": float,
        "l1_ratio": float,
        "C": float,
        "learning_rate": float,
        
        # Boolean Types
        "fit_intercept": lambda x: str(x).lower() == 'true',
    }

    transformed = {}

    for key, value in raw_args.items():
        # Handle 'null' or None (e.g., max_depth: null)
        if value is None or str(value).lower() == 'null':
            transformed[key] = None
            continue

        # If we have a rule for this specific key, use it
        if key in casting_rules:
            try:
                transformed[key] = casting_rules[key](value)
            except (ValueError, TypeError):
                # Fallback: if casting fails, keep original (prevents crash)
                transformed[key] = value
        else:
            # Categorical/Select Types (e.g., kernel: "rbf", weights: "uniform")
            # These are already strings, so no conversion needed
            transformed[key] = value

    return transformed

def to_native(obj):
    if hasattr(obj, 'item'): # Handles NumPy scalars (float64, int64)
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [self.to_native(i) for i in obj]
    if isinstance(obj, dict):
        return {k: self.to_native(v) for k, v in obj.items()}
    return obj

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

    def load_data(self, file_name, pipeline_name):
        base_path = f"./transformed-data/{file_name}/{pipeline_name}"
        try:
            # Load everything as DataFrames. No squeezing yet.
            X_train = pd.read_parquet(f"{base_path}/X_train.parquet")
            X_test = pd.read_parquet(f"{base_path}/X_test.parquet")
            Y_train = pd.read_parquet(f"{base_path}/Y_train.parquet")
            Y_test = pd.read_parquet(f"{base_path}/Y_test.parquet")
        except Exception as error:
            print("Could not read paraquet file")
            print(error)
            raise error
        
        return (X_train, X_test, Y_train, Y_test)
    

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
        base_path = f"./models/{file_name}/{pipeline_name}"
        model_name = model_name + "_"+experiment_name + "_" + hashlib.sha256(json.dumps(arguments,sort_keys=True).encode()).hexdigest()
        os.makedirs(base_path,exist_ok=True)
        joblib.dump(model,f"{base_path}/{model_name}.pk1")
    
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
    
    async def save_experiment(self, payload):
        selected_model = payload['model_name']
        experiment_name = payload['experiment_name']
        file_name = payload['file_name']
        pipeline_name = payload['pipeline_name']
        arguments = payload['arguments'] or {}
        metrics = payload['metrics'] or []
    
        base_path = f"experiments/{file_name}/{pipeline_name}"
        os.makedirs(base_path, exist_ok=True)
        
        file_path = f"{base_path}/{experiment_name}.json"
        
        if Path(file_path).exists():
            raise Exception("An experiment with this name already exists")
        
        saving_data = {
            'file_name': file_name,
            'selected_model': selected_model,
            'experiment_name': experiment_name,
            'pipeline_name': pipeline_name,
            'arguments': arguments,
            'metrics': metrics,
            'current_state': 'created',
        }
        
        with open(file_path, "w") as f:
            f.write(json.dumps(saving_data, sort_keys=True))

        return {
            'experiment_name': experiment_name,
            'pipeline_name': pipeline_name,
            'file_name': file_name,
            'current_state': 'created',
            'arguments':arguments,
            'selected_model':selected_model,
            'metrics':metrics
        }

    async def fit_model(self, payload):
        print(payload,"This is gonna used to ")
        experiment_name = payload['experiment_name']
        file_name = payload['file_name']
        pipeline_name = payload['pipeline_name']
        

        # 1. Validation
        train_path = Path(f"transformed-data/{file_name}/{pipeline_name}/X_train.parquet")
        if not train_path.exists():
            print("Training data not found")
            raise Exception("Data must be transformed via pipeline to train for now")

        experiment_path = Path(f"experiments/{file_name}/{pipeline_name}/{experiment_name}.json")
        if not experiment_path.exists():
            print("Experiment path not found")
            raise Exception("Experiment must be saved to be used")

        # 2. Load Config
        print("Could not rea dhte file")
        with open(experiment_path, "r") as f:
            experiment_info = json.loads(f.read())
        print("read the files ",experiment_info)
        # 3. Setup Model and Data
        # Ensure your get_model_import is actually returning a class instance
    
        model = self.get_model_import(experiment_info['selected_model'],**transform_arguments(experiment_info.get('arguments',{})))
        print("load the model")
        # X_train is a DataFrame, Y_train is likely a DataFrame (from load_data)
        # console.log("Trying to load the paraquet files")
        X_train, X_test, Y_train, Y_test = self.load_data(file_name, pipeline_name)
        print("loaded the data")
        # --- CRITICAL FIX ---
        # Scikit-learn models usually want Y as a 1D array (N,), not a 2D DataFrame (N, 1)
        if isinstance(Y_train, pd.DataFrame):
            y_train_fit = Y_train.iloc[:, 0].values  # Convert to 1D numpy array
        else:
            y_train_fit = Y_train

        try:
            # 4. Train
            # Check if X_train has the columns we expect
            print(f"Fitting model with features: {X_train.columns.tolist()}")
            model.fit(X_train, y_train_fit)
            
            # 5. Save Model
            self.save_model(model, file_name, pipeline_name, experiment_name, experiment_info['selected_model'], experiment_info.get('arguments', {}))

            # 6. Update Metadata
            experiment_info['current_state'] = 'fitted'
            with open(experiment_path, "w") as f:
                # Using json.dump is safer than f.write(json.dumps)
                json.dump(experiment_info, f, sort_keys=True, indent=4)

            return {
                'experiment_name': experiment_name,
                'fitted': True,
                'pipeline_name': pipeline_name,
                'file_name': file_name
            }

        except Exception as e:
            # If it fails here, we return the actual error string
            print(f"Error during model.fit: {str(e)}")
            raise Exception(f"Fitting failed: {str(e)}")




    def get_model_import(self,model_name,**kwargs):


# https://leetcode.com/problems/generate-parentheses/
        print("MODELS ",model_name,MODEL_REGISTRY.get(model_name,{}))
        module_path,class_name = MODEL_REGISTRY[model_name]['import']
        print("found the model ",module_path,class_name)
        module = import_module(module_path)
        model_class = getattr(module,class_name)

        model = model_class(**kwargs)

        return model
    

    async def evaluate_model(self, payload):
        # 1. Extract payload data
        experiment_name = payload['experiment_name']
        file_name = payload['file_name']
        pipeline_name = payload['pipeline_name']
        

        # 2. Path Validation
        experiment_path = f"experiments/{file_name}/{pipeline_name}/{experiment_name}.json"
        if not Path(experiment_path).exists():
            raise Exception("Experiment metadata not found. Save the experiment first.")

        # 3. Load Experiment Metadata
        with open(experiment_path, "r") as f:
            experiment_info = json.loads(f.read())

        # 4. Construct Model Path (matching your fit_model logic)
        arguments = experiment_info.get('arguments', {})
        arg_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        model_name = f"{experiment_info['selected_model']}_{experiment_name}_{arg_hash}.pk1"
        model_path = f"./models/{file_name}/{pipeline_name}/{model_name}"

        if not Path(model_path).exists():
            raise Exception(f"Model file not found at {model_path}. Did you run fit_model?")

        # 5. Load Model and Transformed Data
        # X_test will contain ['Age', 'Gender_Female', 'Gender_Male']
        model = joblib.load(model_path)
        _, X_test, _, Y_test = self.load_data(file_name, pipeline_name)
        print(X_test,"this is where")
        # 6. Predict
        # We pass X_test directly because it's already aligned with what the model learned
        Y_pred = model.predict(X_test)

        # 7. Calculate User-Selected Metrics
        evaluations = []
        
        # Ensure Y_test is a 1D array/series for metric functions
        # Parquet loads as a DataFrame, but metrics expect (n_samples,)
        y_true = Y_test.iloc[:, 0] if isinstance(Y_test, pd.DataFrame) else Y_test

        for metric_name in experiment_info.get('metrics', []):
            if metric_name in METRICS:
                try:
                    metric_fn = METRICS[metric_name]
                    score = metric_fn(y_true, Y_pred)
                    
                    evaluations.append({
                        'metric': metric_name, 
                        'result': to_native(score) # Ensure JSON serializable
                    })
                except Exception as e:
                    print(f"Error calculating {metric_name}: {e}")
                    evaluations.append({
                        'metric': metric_name, 
                        'result': "Error during calculation"
                    })

        # 8. Update Experiment State
        experiment_info['current_state'] = 'evaluated'
        experiment_info['last_evaluations'] = evaluations
        
        with open(experiment_path, "w") as f:
            json.dump(experiment_info, f, sort_keys=True, indent=4)

        # 9. Return Results to Go Backend
        return {
            'experiment_name': experiment_name,
            'evaluations': evaluations,
            'pipeline_name': pipeline_name,
            'file_name': file_name,
            'status': 'success'
        }