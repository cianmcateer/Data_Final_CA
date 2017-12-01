"""Take data from CSV file and restructure data into desirable form and insert into 'students' collection"""
from pymongo import MongoClient

try:
    """Connect to mongod server"""
    client = MongoClient('localhost:27020')
except:
    print("Could not connect to mongo server")

"""Set database and collection we are inserting to"""
db = client.final_data
collection = db.students

path = "/Users/CianMcAteer/year_3/semester_1/data_final_ca/FinalProjectData1718.csv"

"""Put all csv data into a list of strings"""
lines = [line for line in open(path)]
print(lines[0])

for i in range(1, len(lines)):
    """
    Loop through csv lines (excluding row names)
    """
    attributes = lines[i].strip().split(";")

    tests = [
        {'lab_1' : int(attributes[1])},
        {'christmas_test' : int(attributes[2])},
        {'lab_2' : int(attributes[3])},
        {'easter_test' : int(attributes[4])},
        {'lab_3' : int(attributes[5])}
    ]
    student = {}

    student["student_id"] = int(attributes[0])
    student["tests"] = tests
    student["part_time_job"] = attributes[6] == '1'
    student["final_grade"] = int(attributes[7])

    # Insert student object into student collection
    # Value is parsed from python dict to JSON string automatically
    try:
        result = db.students.insert_one(student)
    except:
        print("Error inserting CSV data")
print("Data has been successfully added to server!")
