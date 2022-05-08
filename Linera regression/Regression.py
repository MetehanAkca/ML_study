import matplotlib.pyplot
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle
from matplotlib import style
#read data and define features(X) vs target(y)
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = np.array(data.drop(predict,axis=1))
y = np.array(data[predict])
#Load model if possible,if a current model deos not exist as a pickle file, fit model from scratch and
#keep fitting a new until a threshold level of accuracy (0.9) is reached.Note that initial data is split
#into two groups,namely, training and test groups respectively for both features and the target. finally,save the model
#as a pickle file.Its important to note that although we have saved our model and defined a random state to select
#the same training data every time for our model;the fitting process still uses some degree of randomness while training
#the model,hence each iteration using this model will still give different accuracies(scores)
try:
    pickle_in = open("studentmodel.pickle","rb")
    linear = pickle.load(pickle_in)
    acc = linear.score(x_test, y_test)
except:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1,random_state=1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

while acc<0.9:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
if acc>=0.9:
     print(f"Accuracy is {acc}")
     with open("studentmodel.pickle","wb") as f:
        pickle.dump(linear,f)

#Finally we print the coefficients of features and print mean absolute error and plot the data.
print(f"Co:{linear.coef_} intercept:{linear.intercept_}")


predictions = linear.predict(x_test)
mae = []
for i in range(len(predictions)):
    pred = predictions[i]
    data_point = x_test[i]
    real_value = y_test[i]
    absolute_error = abs(real_value-pred)
    mae.append(absolute_error)
    print(f"The prediction {pred} was made for {data_point} "
          f"has a real value of {real_value} with absolute error of {absolute_error}")
print(f"the mean absolute error is {np.mean(mae)}")

style.use("ggplot")

features = [label for label in data.columns if label!=predict]
figure, axis = pyplot.subplots(5,1)
for row in range(len(features)):
    axis[row].scatter(data[features[row]],data[predict])
    axis[row].set_title(f"{features[row]}")

pyplot.show()