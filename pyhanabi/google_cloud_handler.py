import os

from google.cloud import storage

class GoogleCloudHandler:
    def __init__(self, project, user):
        self._project = project
        self._user = user
        self._client = storage.Client(project=self._project)
        self._bucket = self._client.get_bucket(self._project + "-data")

    def assert_file_doesnt_exist(self, gc_path):
        path = os.path.join(self._user, gc_path)
        blobs = list(self._client.list_blobs(self._bucket, prefix=path))
        assert len(blobs) == 0, f"Google Cloud Error: Path {path} already exists."

    def upload(self, file_path, gc_path):
        if not os.path.exists(file_path):
            return

        blob_name = os.path.join(self._user, gc_path)
        blob = self._bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

    def list_user_blobs(self):
        for blob in self._client.list_blobs(self._bucket):
            if self._user in str(blob.name):
                print(blob)

