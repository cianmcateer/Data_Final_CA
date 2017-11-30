from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def read_menu(path):
    menu = [line for line in open(path)]
    for m in menu:
        print(m)

def extract_json():
    try:
        """Connect to mongod"""
        client = MongoClient('localhost')
        print("Succesful Connection to student database!")
    except:
        print("Could not connect to mongo server")

    """Set database and collection we are going to extract from"""
    db = client.final_data
    students = db.students

    """Return all documents in students collection"""
    return students.find({})

students = extract_json()

student_ids = []
has_job = []
tests = []
final_grades = []

def to_list():
    for s in students:
        student_ids.append(s["student_id"])
        has_job.append(s["part_time_job"])
        exams = s["tests"]
        tests.append([exams[0]["lab_1"], exams[1]["christmas_test"], exams[2]["lab_2"], exams[3]["easter_test"], exams[4]["lab_3"]])
        final_grades.append(s["final_grade"])

to_list()



"""
Part 1: Carry out an initial analysis of the data set.
This should include, the identification and treatment of any outliers/missing data,
the creation of appropriate visualisations and the generation of covariance & correlation statistics.
"""

"""Dealing with outliers"""

def replace_outliers(data):
    """
    Calculate first and third_quartile using numpy.percentile
    loop through list and replace all outliers with mean of list
    """
    output = deepcopy(data)

    first_quartile = np.percentile(output, 25)
    third_quartile = np.percentile(output, 75)

    outlier_indexes = [i for i, x in enumerate(output) if x < first_quartile or x > third_quartile]

    replacement_mean = round(np.mean(output))
    for i in outlier_indexes:
        output.pop(i)
        output.insert(i, replacement_mean)

    return output

grades_minus_outliers = replace_outliers(final_grades)

lab_1_m = np.mean([tests[i][0] for i in range(len(tests))])
christmas_test_m = np.mean([tests[i][1] for i in range(len(tests))])
lab_2_m = np.mean([tests[i][2] for i in range(len(tests))])
easter_test_m = np.mean([tests[i][3] for i in range(len(tests))])
lab_3_m = np.mean([tests[i][4] for i in range(len(tests))])
means = [lab_1_m, christmas_test_m, lab_2_m, easter_test_m, lab_3_m]

tests_mo = [[tests[i][j] for j in range(len(tests[i]))] for i in range(len(tests))]

mean1 = np.mean([tests[i][0] for i in range(len(tests))])

lab_1_no = replace_outliers([tests[i][0] for i in range(len(tests))])
christmas_test_no = replace_outliers([tests[i][1] for i in range(len(tests))])
lab_2_no = replace_outliers([tests[i][2] for i in range(len(tests))])
easter_test_no = replace_outliers([tests[i][3] for i in range(len(tests))])
lab_3_no = replace_outliers([tests[i][4] for i in range(len(tests))])

tests_no = []
for i in range(len(tests)):
    tests_no.append([lab_1_no[i], christmas_test_no[i], lab_2_no[i], easter_test_no[i], lab_3_no[i]])

menu = True


has_job_grades = [final_grades[i] for i in range(len(tests)) if has_job[i]]
no_job_grades = [final_grades[i] for i in range(len(tests)) if not has_job[i]]

has_job_tests = [tests[i] for i in range(len(tests)) if has_job[i]]
no_job_tests = [tests[i] for i in range(len(tests)) if not has_job[i]]

average_test_results = [round(np.mean(tests[i])) for i in range(len(tests))]

print((average_test_results))
def central_tendancy(message,func, data):
    print(message + " " + str(func(data)))

def pearson(x, y):
    """
    Returns pearson coeffcient using numpy
    """
    return np.corrcoef(x, y)[0][1]

while menu:
    read_menu("menus/main_menu.txt")
    choice = input("Please choose")

    if choice == 0:
        menu = False
        print("Goodbye!")

    elif choice == 1:
        """Part 1"""
        """Plots"""
        read_menu("menus/graphs.txt")
        choose_plot = input("Please choose plot")
        def scatter_data(x, y, title):
            plt.title(title)
            plt.xlabel("Scatter Plot: Final Grades vs Tests")
            plt.scatter(x, y, alpha=0.3)
            plt.grid()
            plt.show()

        exams = ["Lab 1", "Christmas Test", "Lab 2", "Easter Test", "Lab 3"]
        if choose_plot == 1:
            for j in range(len(tests[0])):
                scatter_data(final_grades, [tests[i][j] for i in range(len(tests))], "Final grade vs " + str(exams[j]))
                scatter_data(replace_outliers(final_grades), replace_outliers([tests[i][j] for i in range(len(tests))]), "Final grade vs " + str(exams[j]) + "(No outliers)")

        def line_graph(x, y, title):
            plt.plot(x)
            plt.plot(y)
            plt.grid()
            plt.title(title)
            plt.show()

        if choose_plot == 2:
            for j in range(len(tests[0])):
                line_graph(final_grades,[tests[i][j] for i in range(len(tests))], "Final grades vs " + str(exams[j]))
                line_graph(replace_outliers(final_grades),replace_outliers([tests[i][j] for i in range(len(tests))]), "Final grades vs " + str(exams[j] + "(No outliers)"))

        if choose_plot == 3:
            line_graph(has_job_grades, no_job_grades, "Final grades Employed vs Unemployed")
            central_tendancy("Highest grade for Employed students",max, has_job_grades)
            central_tendancy("Highest grade for unemployed students",min, no_job_grades)

            print("Average grade for employed students", np.mean(has_job_grades))
            print("Average grade for unemployed students", np.mean(no_job_grades))

            central_tendancy("Lowest grade for Employed students",min,has_job_grades)
            central_tendancy("Lowest grade for Unemployed students",min,no_job_grades)

            print("Pearson correlation")
            print("Mean of previous tests vs final exam " + pearson(average_test_results, final_grades))


    else:
        print("Invalid graph input")
