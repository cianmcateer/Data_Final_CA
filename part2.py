from pymongo import MongoClient

def extract_json():
    try:
        """Connect to mongod"""
        client = MongoClient('localhost')
    except:
        print("Could not connect to mongo server")

    """Set database and collection we are going to extract from"""
    db = client.final_data
    students = db.students

    """Return all documents in students collection"""
    return students.find({})

students = extract_json()

for s in students:
    print(s)
