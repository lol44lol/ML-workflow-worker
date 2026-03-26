import pandas as pd
import numpy as np
import event_types
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import json
import pathlib
import os
import events
import websocket
import scalers
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
    
    def close_dataset(self,payload):
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

    async def transform_data(self, payload):
        pathlib.Path("encoders").mkdir(exist_ok=True)
        pathlib.Path("scalers").mkdir(exist_ok=True)
        file_name = payload['file_name']       
        pipeline_name = payload['pipeline_name']
        

        dataset = self.datasets.get(file_name)
        self.guard(dataset)

        pipeline = self.get_pipeline(file_name,pipeline_name)

        if not pipeline:
            raise PipeLineDoesNotExist("Pipeline needs to be created first")

        categorical_columns = pipeline['categorical_columns']
        numerical_columns = pipeline['numerical_fields']
        predict_label = pipeline['predict_label']
        remove_fields = pipeline['remove_fields']
        scaler_name = pipeline['scaler']

        df = dataset.copy()
        df.dropna(inplace=True)
        Y = df[predict_label]

        df.drop(columns=remove_fields, inplace=True)
        df.drop(columns=[predict_label], inplace=True)

        X_train, X_test, Y_train, Y_test = train_test_split(
            df, Y, random_state=42
        )

        await socket.sendMessage({"file_name":file_name,
                            "pipeline_name":pipeline_name},
                            events.PIPELINE_ONE_HOT_ENCODER_INITIALIZING)
        # time.sleep(5)
        X_train_final = X_train
        X_test_final = X_test
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(X_train[categorical_columns])
            await socket.sendMessage({"file_name":file_name,
                                "pipeline_name":pipeline_name},
                                events.PIPELINE_ONE_HOT_ENCODER_FITTED)
            # time.sleep(5)
            
            self.save_encoder(encoder,file_name,pipeline_name)
            await socket.sendMessage({"file_name":file_name,
                                "pipeline_name":pipeline_name},
                                events.PIPELINE_ONE_HOT_ENCODER_SAVED)
            pipeline['encoder'] = encoder

             # time.sleep(5)
            X_train_cat = encoder.transform(X_train[categorical_columns])
            X_test_cat  = encoder.transform(X_test[categorical_columns])

            cat_features = encoder.get_feature_names_out(categorical_columns).tolist()

            X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_features, index=X_train.index)
            X_test_cat_df  = pd.DataFrame(X_test_cat, columns=cat_features, index=X_test.index)

            X_train_final = pd.concat([X_train[numerical_columns], X_train_cat_df], axis=1)
            X_test_final  = pd.concat([X_test[numerical_columns], X_test_cat_df], axis=1)

            X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)
        

        scaler_class = SCALERS.get(scaler_name)
        if scaler_class:
            scaler = scaler_class()
            scaler.fit(X_train_final)
            
            X_train_final = scaler.transform(X_train_final)
            X_test_final = scaler.transform(X_test_final)

            self.save_scaler(scaler,file_name,pipeline_name)




        self.transformed_data[self.get_pipeline_full_name(file_name,pipeline_name)] = {
            "X_train": X_train_final,
            "X_test": X_test_final,
            "Y_train": Y_train,
            "Y_test": Y_test,
        }

        base_path = f"./transformed-data/{file_name}/{pipeline_name}"
        os.makedirs(base_path,exist_ok=True)
        X_train_final.to_parquet(f"{base_path}/X_train.parquet")
        Y_train.to_frame().to_parquet(f"{base_path}/Y_train.parquet")
        X_test_final.to_parquet(f"{base_path}/X_test.parquet")
        Y_test.to_frame().to_parquet(f"{base_path}/Y_test.parquet")



        return {"transformed": True,"file_name":file_name,"pipeline_name":pipeline_name}

    def get_dataset(self,filename,pipeline_name):
        return self.transformed_data.get(self.get_pipeline_full_name(filename,pipeline_name))

    