from dotenv import load_dotenv, find_dotenv

from vdb.vector_store import VectorStore

load_dotenv(find_dotenv())
import os
import shutil
from data_utils.video import MovVideoLoader

VIDEO_DIR = os.environ.get("VIDEO_DIR", "videos")
WORKING_DIR = os.environ.get("WORKING_DIR", "data")


def main():
    data_dir = os.path.abspath(os.path.join(WORKING_DIR, VIDEO_DIR))
    video_loader = MovVideoLoader()
    videos_data = video_loader.load(data_dir)
    vdb_dir = os.path.join(WORKING_DIR, "qdrant_data")
    if os.path.exists(vdb_dir):
        shutil.rmtree(vdb_dir)
    COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
    vector_store = VectorStore(COLLECTION_NAME)
    vector_store.add_documents(videos_data)
    vector_store.persist()
    print("Done!")


if __name__ == "__main__":
    main()
