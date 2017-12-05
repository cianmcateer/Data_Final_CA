from pymongo import MongoClient
import numpy as np
from numpy import dot
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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


"""Dealing with outliers"""

def replace_outliers(data):
    """
    Calculate first and third_quartile using numpy.percentile
    loop through list and replace all outliers with mean of list
    """
    output = deepcopy(data)

    first_quartile = np.percentile(output, 25)
    third_quartile = np.percentile(output, 75)

    #outlier_indexes = [i for i, x in enumerate(output) if x < first_quartile or x > third_quartile]

    replacement_mean = np.mean(output)
    for i in range(len(output)):
        if output[i] < first_quartile or output[i] > third_quartile:
            output[i] = replacement_mean

    return output

grades_minus_outliers = replace_outliers(final_grades)

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



def pearson(x, y):
    """
    Returns pearson coeffcient using numpy
    """
    return np.corrcoef(x, y)[0][1]

def r_squared(x):
    return x**2

def pearson_results(coef):
    if coef == 1:
        print("Perfect positve correlation")
    elif coef > 0.75:
        print("Very strong positve correlation")
    elif coef > 0.5:
        print("Strong positve correlation")
    elif coef > 0:
        print("Positve correlation")
    elif coef > -0.5:
        print("Negative correlation")
    elif coef > -0.75:
        print("Strong Negative correlation")
    elif coef == -1:
        print("Perfect Negative coeffcient")
    else:
        print("Invalid value")

while menu:
    read_menu("menus/main_menu.txt")
    choice = input("Please choose")
    exams = ["Lab 1", "Christmas Test", "Lab 2", "Easter Test", "Lab 3"]
    
    if choice == 0:
        menu = False
        print("Goodbye!")


    elif choice == 1:
        """Part 1"""

        """
        Part 1: Carry out an initial analysis of the data set.
        This should include, the identification and treatment of any outliers/missing data,
        the creation of appropriate visualisations and the generation of covariance & correlation statistics.
        """

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

        def line_graph(x, y, title, x_label, y_label):
            plt.plot(x, label=x_label)
            plt.plot(y, label=y_label)
            plt.grid()
            plt.legend(loc='upper left')
            plt.title(title)
            plt.show()

        if choose_plot == 2:
            for j in range(len(tests[0])):
                line_graph(final_grades,[tests[i][j] for i in range(len(tests))], "Final grades vs " + exams[j], "Final grades", exams[j])
                line_graph(replace_outliers(final_grades),replace_outliers([tests[i][j] for i in range(len(tests))]), "Final grades vs " + exams[j] + "(No outliers)", "Final grades", exams[j])

        if choose_plot == 3:
            print(max(final_grades))
            line_graph(has_job_grades, no_job_grades, "Final grades Employed vs Unemployed", "Employed", "Unemployed")
            print("Highest grade for Employed students", max(has_job_grades))
            print("Highest grade for unemployed students", min(no_job_grades))

            print("Mean for employed students", np.mean(has_job_grades))
            print("Mean for unemployed students", np.mean(no_job_grades))

            print("Standard deviation for employed students", np.std(has_job_grades))
            print("Standard deviation for unemployed students", np.std(no_job_grades))

        if choose_plot == 4:
            for j in range(len(tests[j])):
                scatter_data(no_job_grades, [no_job_tests[i][j] for i in range(len(no_job_tests))], "Unemployed grades vs Unemployed " + exams[j])

        if choose_plot == 5:
            line_graph(has_job_grades, no_job_grades, "Employed v Unemployed student final grades", "Employed", "Unemployed")
            for j in range(len(has_job_tests[0])):
                line_graph([has_job_tests[i][j] for i in range(len(has_job_tests))], [no_job_tests[i][j] for i in range(len(no_job_tests))], "Employed v Unemployed students " + exams[j], "Employed", "Unemployed")

        if choose_plot == 6:


            print("Average grade for employed students", np.mean(has_job_grades))
            print("Average grade for unemployed students", np.mean(no_job_grades))

            print("Lowest grade for Employed students",min(has_job_grades))
            print("Lowest grade for Unemployed students",min(no_job_grades))

            print(np.cov(final_grades, [tests[i][0]for i in range(len(tests))]) )

            print("Pearson correlation")
            print("Correlation tests vs final exam " + str(pearson(average_test_results, final_grades)))
            print(pearson_results(pearson(average_test_results, final_grades)))

    elif choice == 2:

        """
        2. Conduct a simple linear regression analysis:
        Determine the 'best' simple linear regression model to predict end-of-module Exam grades.
        Explain your model & justify your choice of model.
        """
        print("Linear regression")

        def de_mean(x):
        	x_bar=np.mean(x)
        	return[x_i-x_bar for x_i in x]

        def covariance(x,y):
        	n=len(x)
        	return np.dot(de_mean(x),de_mean(y))/(n-1)

        def correlation(x,y):
        	stdev_x=np.std(x)
        	stdev_y=np.std(y)
        	if stdev_x>0 and stdev_y>0:
        		return covariance(x,y)/stdev_x/stdev_y
        	else:
        		return 0

        def least_squares_fit(x, y):
            beta = correlation(x, y)*np.std(x)*np.std(y)
            alpha = np.mean(y)-beta*np.mean(x)
            return alpha, beta

        for j in range(len(tests[0])):
            alpha, beta = least_squares_fit(final_grades, [tests[i][j] for i in range(len(tests))])
            y=pearson(final_grades,[tests[i][j] for i in range(len(tests))])
            print("correlation between Finalgrade & " + exams[j], y)
            print("rsquared results Finalgrade & " + exams[j],r_squared(y))
            print("Alpha Finalgrade & " + exams[j], alpha)
            print("beta Finalgrade & " + exams[j], beta)
            print("Regression Line: y =", beta, "x +",alpha)
            print("")
            print("")


        def predict_y(alpha,beta,x):
        	return alpha+beta*x

        def predict_x(alpha,beta,y):
        	return (y-alpha)/beta

        def impact_y_deltax(beta,x):
        	return beta*x

        #
        # Residual's
        # y_i = beta x_i + alpha + e_i,......   (e_i's are the errors)
        #

        def error(alpha,beta,x,y):
        	error = y-predict_y(alpha,beta,x)
        	return error

        def de_mean(x):
        	x_bar=np.mean(x)
        	return[x_i-x_bar for x_i in x]

        def sum_of_squared_errors(alpha,beta,x,y):
        	return sum((error(alpha,beta,x_i,y_i)**2 for x_i, y_i in zip(x,y)))

        def sum_of_squares(x):
        	return sum([x_i**2 for x_i in x])

        for j in range(len(tests[0])):
            print(sum_of_squared_errors(alpha,beta,final_grades,[tests[i][j] for i in range(len(tests))]))

        print("")

        def r_squared2(alpha,beta,x,y):
        	return 1-(sum_of_squared_errors(alpha,beta,x,y)/sum_of_squares(de_mean(y)))

        def least_squares_fit_test(x,y,i):
        	beta = correlation(x,y)*np.std(y)/np.std(x)
        	alpha = np.mean(y)-(beta+.1*i)*np.mean(x)
        	return alpha, beta+.1*i

        def scatter_regression_line(x, y, x_line, y_line, title):

            # Scatter plot
            plt.title(title)
            plt.scatter(x, y)
            plt.xlabel(x_line)
            plt.ylabel(y_line)

            # Add least squares regression line
            axes = plt.gca()
            m, b = np.polyfit(x, y, 1)
            x_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
            #Add line to graph
            plt.plot(x_plot, m*x_plot + b, '-')
            plt.show()

        """for j in range(len(tests[0])):
            scatter_regression_line(final_grades, [tests[i][j] for i in range(len(tests))], "Final grades", exams[j], "Final grades vs " + exams[j])
            scatter_regression_line(replace_outliers(final_grades), replace_outliers([tests[i][j] for i in range(len(tests))]), "Final grades (No outliers)", exams[j] + " (No outliers)", "Final grades vs " + exams[j] + " (No outliers)")"""

        for j in range(len(tests[0])):
            print("Sum of squared errors", sum_of_squared_errors(alpha,beta,final_grades,tests[j]))
            print(r_squared2(alpha,beta,final_grades,tests[j]))
            print("")


        """for i in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]:
        	alpha,beta = least_squares_fit_test(num_friends_good,daily_minutes_good,i)
        	print(i,sum_of_squared_errors(alpha,beta,num_friends_good,daily_minutes_good),sum_of_squares(de_mean(daily_minutes_good)))
        	print(i,r_squared(alpha,beta,num_friends_good,daily_minutes_good))
        	alpha,beta = least_squares_fit_test(num_friends_good,daily_minutes_good,i)
        	print(-i,sum_of_squared_errors(alpha,beta,num_friends_good,daily_minutes_good))
        	print(-i,r_squared(alpha,beta,num_friends_good,daily_minutes_good))"""

    elif choice == 3:
        print("multiple linear regression")

        """
        3. Conduct a multiple linear regression analysis: Fit a multiple regression line predicting end-of-module Exam grades.
        Justify your choice of features, interpret your model output and using, R2 and p-values,
        determine the usefulness of your model and the significance of each of the factors included.
        """

        print("Using Bootstrap sampling with linear regression")

        # Default x and y sum_of_squarred_errors(v,x=d1,y=d2)
        def predict_y(v,x):
            """
            Performs matrix multiplication on two lists
            """
            return dot(v, x)

        def error_b(v,x,y):
            error = y-predict_y(v,x)
            return error

        def squared_error(v, x, y):
            return error_b(v,x,y)**2

        #
        # Estimate Gradient
        #

        def partial_difference_quotient(f,v,x,y,i,h):
            # Compute the ith partial difference quotient of f at v
            w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
            # adding h to just thee ith element of v
            return (f(w,x,y) - f(v,x,y)) / h

        def estimate_gradient(f, v, x, y, h = 0.00001):
            """ For each element of list passed in calculate the partial difference quotient"""
            return [partial_difference_quotient(f, v,x,y, i, h) for i,_ in enumerate(v)]

        #
        #   Gradient Descent
        #
        def step(v, direction, step_size):
            return [v_i - step_size * direction_i for v_i, direction_i in zip(v,direction)]

        v = [random.randint(-10,10) for i in range(2)]

        def distance_e(v1, v2):
            deviations = [v2_i - v1_i for v2_i, v1_i in zip(v2,v1)]
            sum_of_squares = sum(deviation ** 2 for deviation in deviations)
            return math.sqrt(sum_of_squares)

        #
        #
        # ---------------------------------
        #       Multiple Variable Functions
        # ---------------------------------
        #
        #       Estimating gradients
        #
        #       Pick random starting point
        def stochastic_gradient(x,y):
            """
            Edited function from previous video
            to fit in with bootstrap sample
            """

            random.seed(0)
            v_0 = [random.random() for x_i in x[0]]
            v = v_0
            step_size = 0.001
            no_improvement = 0
            step_0 = 0.001
            min_v = None
            min_value = float("inf")

            #
            #   Main loop
            #
            count = 0
            while no_improvement < 100:

                value = sum(squared_error(v,x_i,y_i) for x_i,y_i in zip(x,y))

                if value < min_value:   # Found a new minimum
                    min_v = v
                    min_value = value
                    no_improvement = 0
                    step_size = step_0

                else:
                    no_improvement += 1
                    step_size *= 0.9

                indexes = np.random.permutation(len(x))

                for i in indexes:
                    x_i = x[i]
                    y_i = y[i]

                    gradient_i = estimate_gradient(squared_error,v,x_i,y_i)
                    v = step(v, gradient_i, step_size)
            """Function now returns result instead of printing it"""
            return min_v

        def bootstrap_sample(data):
            """
            Since we are dealing with a 2d list,
            We must convert zip into a 1d list
            """
            list_data = list(data)
            return [random.choice(list_data) for _ in list_data]

        def bootstrap_statistic(x,y, stats_fn, num_samples):
            """
            Evaluates stats_fn on num_samples bootstrap samples from data
            """

            stats = []
            for i in range(num_samples):
                data = zip(x, y)
                sample_data = bootstrap_sample(data) # Unzip list
                x_sample, y_sample = zip(*sample_data) # Assign
                x_list = list(x_sample)
                y_list = list(y_sample)
                stat=stats_fn(x_list, y_list)
                stats.append(stat)
            return stats

        # NOTE: change 20 to a lower value to load faster
        sample_stotatics = bootstrap_statistic(tests, final_grades, stochastic_gradient, 1)

        def multiple_linear_regression(test1, christmas_test, lab2, easter_test, has_job):

            linear_regressions = []
            for i in range(len(sample_stotatics)):
                regression = (sample_stotatics[i][0]*test1) + (sample_stotatics[i][1]*christmas_test) + (sample_stotatics[i][2]*lab2), (sample_stotatics[i][3]*easter_test)
                linear_regressions.append(regression)
            return linear_regressions
        for i in range(len(sample_stotatics)):
            print(multiple_linear_regression(44, 66, 33, 55, False))

    elif choice == 4:

        """
        Carry out a k-means cluster analysis on the data.
        Justify your choice of k and detail how this choice was made.
        """

        def to_list(x, y):
            """
            Returns 2d list of each element x and y
            """
            return [[x[i], y[i]] for i in range(len(x))]

        def get_column(m, j):
            return [m_i[j] for m_i in m]
        employed_v_unemployed = to_list([final_grades[i] for i in range(len(final_grades)) if has_job[i]], [final_grades[i] for i in range(len(final_grades)) if not has_job[i]])

        def generate_k_means(data, clusters=4):
            print("K means clustering")
            K_means = KMeans(n_clusters=clusters) # Define number of clusters

            K_means.fit(data)
            cluster_assignment = K_means.predict(data)    # Extracts
            print("Shows which cluster values are assigned to")

            cluster0 = []
            cluster1 = []
            cluster2 = []
            cluster3 = []

            for k in range(len(cluster_assignment)):
                if cluster_assignment[k] == 0:
                    cluster0.append(data[k])
                if cluster_assignment[k] == 1:
                    cluster1.append(data[k])
                if cluster_assignment[k] == 2:
                    cluster2.append(data[k])
                if cluster_assignment[k] == 3:
                    cluster3.append(data[k])

            x_cluster0 = get_column(cluster0, 0)
            y_cluster0 = get_column(cluster0, 1)

            x_cluster1 = get_column(cluster1, 0)
            y_cluster1 = get_column(cluster1, 1)

            x_cluster2 = get_column(cluster2, 0)
            y_cluster2 = get_column(cluster2, 1)

            x_cluster3 = get_column(cluster3, 0)
            y_cluster3 = get_column(cluster3, 1)

            plt.scatter(x=x_cluster0, y=y_cluster0, color='green')
            plt.scatter(x=x_cluster1, y=y_cluster1, color='red')
            plt.scatter(x=x_cluster2, y=y_cluster2, color='blue')
            plt.scatter(x=x_cluster3, y=y_cluster3, color='black')
            plt.show()

        generate_k_means(employed_v_unemployed)

        for j in range(len(tests[0])):
            # Compare final grades to each individual test
            final_grades_to_test = to_list(final_grades, [tests[i][j] for i in range(len(tests))])
            generate_k_means(final_grades_to_test)


    elif choice == 5:
        print("Principle componant analysis")
        """
        5. Use principal component analysis to reduce the dimensionality in your data set,
        identify the first & second principal components of
        your data set and create appropriate visualisations of your clusters.
        """

    else:
        print("Invalid input, please try again")
