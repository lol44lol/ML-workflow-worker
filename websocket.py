import websockets
import json

class WebSocket:


    def __init__(self):
        self.ws = None
        pass

    async def connect(self):
        self.ws = await websockets.connect("ws://localhost:8080/api/v1/connections/init-python")

    async def sendMessage(self,data,event,process_id=0,hasError=False,error=""):
        await self.ws.send(json.dumps(
            {
                "data":data,
                "event":event,
                "has_error":hasError,
                "error":error,
                "process_id":process_id
            }
        ))

    async def onMessage(func):
        pass

websocket = WebSocket()
