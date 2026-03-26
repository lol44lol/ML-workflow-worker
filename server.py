import grpc
from concurrent import futures
import os

import file_pb2
import file_pb2_grpc


class FileService(file_pb2_grpc.FileServiceServicer):

    def SendFileInfo(self, request, context):

        file_path = request.file_path

        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            status = "file_found"
        else:
            size = 0
            status = "file_not_found"

        return file_pb2.FileResponse(
            status=status,
            file_size=size
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    file_pb2_grpc.add_FileServiceServicer_to_server(FileService(), server)

    server.add_insecure_port("[::]:50051")
    server.start()
    print("Python gRPC server running on :50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()