from pymongo import MongoClient

client = None
db = None

def connect_db():
    global client, db
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ml_database"]
