import asyncio
import json
import websocket
import datasets
import event_types
import events
import models

datasets = datasets.Datasets()
models = models.Models()

WS = None

def wrapper(func,eventID):

    
    async def wrap(event):
        payload = event['data']
        try:
            output = await func(payload)
            print("got the output ",output)
            returningOutput = {
                'data':output,
                'process_id' :event['process_id'],
                'event':eventID,
                'error':"",
                'has_error':False
            }
         

            return returningOutput
        except Exception as error:
            print(error)
            return {
                "error":str(error),
                "has_error":True,
                'process_id':event['process_id'],
                'data':{},
                'event':eventID
            }

    return wrap
    




async def main():
    
    EVENTS = {
    'DATASET_LOAD':wrapper(datasets.load_dataset,"DATASET_LOADED"),
    'DATASET_CLOSE':wrapper(datasets.close_dataset,"DATASET_CLOSED"),
    'DATASET_HEAD':wrapper(datasets.get_head,"DATASET_HEAD_INFO"),
    'DATASET_READ_SAMPLE':wrapper(datasets.read_num_of_rows,"DATASET_READ_COMPLETED_SAMPLE"),
    'DATASET_BUILD_PIPELINE':wrapper(datasets.build_pipeline,"DATASET_BUILT_PIPELINE"),
    'DATASET_TRANSFORM_DATA':wrapper(datasets.transform_data,"DATASET_TRANSFORMED_DATA"),
    "MODEL_BUILD_NEW_MODEL":wrapper(models.build_model,events.MODEL_BUILT_FINISHED)
}

    socket = websocket.websocket
    await socket.connect()
    print("Connected")
    try:
        while True:
            info = await socket.ws.recv()
            try:
                info = json.loads(info)
                currentEvent = EVENTS[info['event']]
                if not currentEvent:
                    continue
                output = await currentEvent(info)
                stringified_output = json.dumps(output)
                print(stringified_output)
                await socket.ws.send(stringified_output)
            except Exception as error:
                print(error)
        # Wait forever until connection closes
        await ws.wait_closed()
    finally:
        print("Connection closed")

asyncio.run(main())