import pandas as pd
import numpy as np
import event_types
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import joblib
import json
import pathlib
import os
import events
import websocket
import scalers
from pathlib import Path
# import main

SCALERS = scalers.SCALERS
socket = websocket.websocket

class DataSetLoadingFailed(Exception):
    def __init__(self, *args):
        super().__init__(*args)
    pass

class DataSetNeedToLoadFirst(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class PipeLineAlreadyExist(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class PipeLineDoesNotExist(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class Datasets:

    def __init__(self):
        self.datasets = {
        }

        self.pipelines = {

        }

        self.transformed_data = {

        }


        pass


    def guard(self,dataset):
        if  dataset is None:
            raise DataSetNeedToLoadFirst("Dataset needs to be loaded first")

    async def load_dataset(self,payload):
        try:
           self.datasets[payload['file_name']] = pd.read_csv(payload['file_path'])
           self.datasets[payload['file_name']] = self.datasets[payload['file_name']].replace({np.nan: None})
           return {}
        except Exception as error:
            raise DataSetLoadingFailed("Dataset could not be loaded")

    async def get_head(self,payload):
        dataset = self.datasets.get(payload['file_name'])
        self.guard(dataset)
        return dataset.head().to_dict()
    
    async def close_dataset(self,payload):
        dataset = self.datasets.get(payload['file_name'])
        self.guard(dataset)
        del self.datasets[payload['file_name']]
        return {"deleted":True}
    
    async def read_num_of_rows(self,payload):
        dataset = self.datasets.get(payload['file_name'])
        self.guard(dataset)
        return dataset.sample(payload['rows']).to_dict()
    
    def get_pipeline_full_name(self,file_name,pipeline_name):
        return file_name + "_"+pipeline_name
    
    async def build_pipeline(self,payload):
        pathlib.Path("pipelines").mkdir(exist_ok=True)

        file_name = payload.get('file_name')
        none_null_value_action = payload.get("none_null_value_action")
        dataset = self.datasets.get(file_name)
        self.guard(dataset)
        
        categorical_columns  = payload['categorical_fields'] or []
        numerical_fields = payload['numerical_fields']
        pipeline_name = payload['pipeline_name']
        predict_label = payload['predict_label']
        remove_fields = payload['remove_fields']
        
        full_pipeline_name = self.get_pipeline_full_name(file_name,pipeline_name)
        pipeline_path = "pipelines/"+full_pipeline_name
        existingPipeline = self.pipelines.get(full_pipeline_name)
        if existingPipeline:
            raise PipeLineAlreadyExist("Pipeline is already exists")
        
        saved_pipeline = self.get_pipeline(file_name,pipeline_name)
        self.pipelines[full_pipeline_name] = saved_pipeline
        if saved_pipeline:
            raise PipeLineAlreadyExist("Pipeline is already exists")
            

        train_df,test_df = train_test_split(dataset,random_state=42)

        
        columns = []
        if categorical_columns:
            encoder = OneHotEncoder()
            encoder.fit(train_df[categorical_columns])
            columns = encoder.get_feature_names_out(categorical_columns).tolist()
        
        self.pipelines[full_pipeline_name] = {
            "categorical_columns":categorical_columns,
            "numerical_fields":numerical_fields,
            "predict_label":predict_label,
            "remove_fields":remove_fields,
            "none_null_value_action":none_null_value_action,
            "encoder":"one-hot-encoder",
            "scaler":payload['scaler']
        }
        pipeLine_content = json.dumps(self.pipelines[full_pipeline_name])
        
        base_path = f"./pipelines/{file_name}/"
        os.makedirs(base_path,exist_ok=True)
        
        with open(f"{base_path}/{pipeline_name}.json","w") as p:
            p.write(pipeLine_content)
        
        return {"after_one_hot_encoding":columns}

    def get_pipeline(self,file_name,pipeline_name):
        full_name = self.get_pipeline_full_name(file_name,pipeline_name)
        exists_pipeline = self.pipelines.get(full_name)
        if  exists_pipeline:
            return exists_pipeline
        
        full_path  = f"./pipelines/{file_name}/{pipeline_name}.json"

        path = pathlib.Path(full_path)
        print("Trying to get the pipeline")
        if path.exists():
            with open(full_path,"r") as f:
                return json.loads(f.read())
        print("No pipeline found")
        return None


    def save_encoder(self,encoder,file_name,pipeline_name):
        pathlib.Path(f"encoders/{file_name}").mkdir(exist_ok=True)
        save_path = f"encoders/{file_name}/{pipeline_name}.pk"
        joblib.dump(encoder,save_path)

    def save_scaler(self,scaler,file_name,pipeline_name):
        pathlib.Path(f"scalers/{file_name}").mkdir(exist_ok=True)
        save_path = f"scalers/{file_name}/{pipeline_name}.pk"
        joblib.dump(scaler,save_path)

    def save_label_encoder(self,encoder,file_name,pipeline_name):
        pathlib.Path(f"label-encoders/{file_name}").mkdir(exist_ok=True)
        save_path = f"label-encoders/{file_name}/{pipeline_name}.pk"
        joblib.dump(encoder,save_path)

    async def transform_data(self, payload):
        # 1. Setup paths and basic variables
        pathlib.Path("encoders").mkdir(exist_ok=True)
        pathlib.Path("scalers").mkdir(exist_ok=True)
        
        file_name = payload['file_name']       
        pipeline_name = payload['pipeline_name']
        
        # Define base_path early to prevent scoping errors
        base_path = f"./transformed-data/{file_name}/{pipeline_name}"

        # 2. Dataset and Pipeline Validation
        dataset = self.datasets.get(file_name)
        self.guard(dataset)

        pipeline = self.get_pipeline(file_name, pipeline_name)
        if not pipeline:
            raise Exception("Pipeline needs to be created first")
        
        # Check if we've already done this work to save time locally
        if Path(f"{base_path}/X_train.parquet").exists():
            return {"transformed": True, "file_name": file_name, "pipeline_name": pipeline_name}

        # 3. Prepare features
        categorical_columns = pipeline.get('categorical_columns', [])
        numerical_columns = pipeline.get('numerical_fields', [])
        predict_label = pipeline['predict_label']
        remove_fields = pipeline.get('remove_fields', [])
        scaler_name = pipeline.get('scaler')

        df = dataset.copy()
        df.dropna(inplace=True)
        Y = df[predict_label]

        # Drop unwanted fields
        cols_to_drop = list(set(remove_fields + [predict_label]))
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Initial Split
        X_train, X_test, Y_train, Y_test = train_test_split(
            df, Y, random_state=42
        )

        await socket.sendMessage({"file_name": file_name, "pipeline_name": pipeline_name},
                                events.PIPELINE_ONE_HOT_ENCODER_INITIALIZING)

        X_train_final = X_train
        X_test_final = X_test

        # 4. Categorical Encoding
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(X_train[categorical_columns])
            
            await socket.sendMessage({"file_name": file_name, "pipeline_name": pipeline_name},
                                    events.PIPELINE_ONE_HOT_ENCODER_FITTED)
            
            self.save_encoder(encoder, file_name, pipeline_name)
            
            await socket.sendMessage({"file_name": file_name, "pipeline_name": pipeline_name},
                                    events.PIPELINE_ONE_HOT_ENCODER_SAVED)
            
            pipeline['encoder'] = encoder

            # Transform
            X_train_cat = encoder.transform(X_train[categorical_columns])
            X_test_cat  = encoder.transform(X_test[categorical_columns])

            # Ensure we have DataFrames with named columns
            if not isinstance(X_train_cat, pd.DataFrame):
                cat_features = encoder.get_feature_names_out(categorical_columns).tolist()
                X_train_cat = pd.DataFrame(X_train_cat, columns=cat_features, index=X_train.index)
                X_test_cat  = pd.DataFrame(X_test_cat, columns=cat_features, index=X_test.index)

            # Merge Numerical + Categorical
            X_train_final = pd.concat([X_train[numerical_columns], X_train_cat], axis=1)
            X_test_final  = pd.concat([X_test[numerical_columns], X_test_cat], axis=1)
            
            # Alignment: Force X_test to have exact same columns as X_train
            X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

        # 5. Scaling (The critical part for preserving feature names)
        scaler_class = SCALERS.get(scaler_name)
        if scaler_class:
            scaler = scaler_class()
            
            # Save column names and indices before scaling
            current_columns = X_train_final.columns
            train_idx = X_train_final.index
            test_idx = X_test_final.index

            scaler.fit(X_train_final)
            
            # Scaling often returns NumPy arrays, losing our column names
            X_train_res = scaler.transform(X_train_final)
            X_test_res = scaler.transform(X_test_final)

            # Reconstruct DataFrames to keep the model from getting confused later
            X_train_final = pd.DataFrame(X_train_res, columns=current_columns, index=train_idx)
            X_test_final = pd.DataFrame(X_test_res, columns=current_columns, index=test_idx)

            self.save_scaler(scaler, file_name, pipeline_name)

        # 6. Label Encoding (Target Variable)
        if Y_train.dtype == "object" or str(Y_train.dtype) == 'category':
            label_encoder = LabelEncoder()
            label_encoder.fit(Y_train)
            
            # Convert to Series to maintain index alignment with X
            Y_train = pd.Series(label_encoder.transform(Y_train), index=X_train_final.index, name=predict_label)
            Y_test = pd.Series(label_encoder.transform(Y_test), index=X_test_final.index, name=predict_label)
            
            self.save_label_encoder(label_encoder, file_name, pipeline_name)

        # 7. Store in memory (Optional: depends on your class usage)
        self.transformed_data[self.get_pipeline_full_name(file_name, pipeline_name)] = {
            "X_train": X_train_final,
            "X_test": X_test_final,
            "Y_train": Y_train,
            "Y_test": Y_test,
        }

        # 8. Persistence (Save to Disk as Parquet)
        os.makedirs(base_path, exist_ok=True)
        
        # Save Features
        X_train_final.to_parquet(f"{base_path}/X_train.parquet")
        X_test_final.to_parquet(f"{base_path}/X_test.parquet")
        
        # Save Labels (Ensure they are DataFrames for Parquet compatibility)
        if isinstance(Y_train, pd.Series):
            Y_train.to_frame().to_parquet(f"{base_path}/Y_train.parquet")
            Y_test.to_frame().to_parquet(f"{base_path}/Y_test.parquet")
        else:
            # Fallback for raw numpy/other types
            pd.DataFrame(Y_train, columns=[predict_label]).to_parquet(f"{base_path}/Y_train.parquet")
            pd.DataFrame(Y_test, columns=[predict_label]).to_parquet(f"{base_path}/Y_test.parquet")

        return {"transformed": True, "file_name": file_name, "pipeline_name": pipeline_name}
    def get_dataset(self,filename,pipeline_name):
        return self.transformed_data.get(self.get_pipeline_full_name(filename,pipeline_name))

  
