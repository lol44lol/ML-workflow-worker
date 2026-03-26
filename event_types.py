from typing import TypedDict

class BaseDict(TypedDict):
    process_id  = int
    event = str
    data = any


class LoadDataset(TypedDict):
    file_name = str
    file_path = str


class LoadDatasetHead (TypedDict):
    file_name = str

class DatasetClose(TypedDict):
    file_name = str