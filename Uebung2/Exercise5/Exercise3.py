
# coding: utf-8

# # Aufgabe 3: Cross Validation und Grid Search

# In dieser Aufgabe geht es um Hyperparameter-Optimierung mittels _Grid-Search_ (deutsch: Gittersuche). Dabei handelt es sich um eine Brute-Force-Suche, die auf einen angegebenen Teil des Hyperparameterraums angewandt wird. Sie wird dabei von einer Performance-Metrik geleitet.

# Die Gittersuche soll in diesem Notebook auf zwei verschiedene Klassifikationsaufgaben angewandt werden, wobei die Anzahl der verwendeten CPU-Kerne variiert werden soll:
# 
# - Iris Flower Dataset
# - Heart Disease UCI Dataset

# Anschließend soll für beide Aufgaben jeweils eine Kurve mit der Suchzeit in Abhängigkeit von der Anzahl an CPU- Kernen geplottet werden. Außerdem soll die Zielmetrik in Abhängigkeit von einem ausgewählten Hyperparameter geplottet werden.

# In [Aufgabe 5](./Exercise5.ipynb) soll diese Aufgabe zudem im Batch-System des SDIL ausgeführt werden. Aus diesem Grund wird in diesem Notebook darauf geachtet, dass es ohne Probleme in ein auführbares Python-Skript konvertiert werden kann.

# ## Bibliotheken Importieren

# Importieren der Bibliotheken. Beim Importieren von `matplotlib` muss darauf geachtet werden, das `agg`-Backend zu verwenden, wenn der Code nicht in einem Notebook ausgeführt wird, da es sonst zu Fehlern kommt.

# In[14]:


## Import Libraries

# Technical
import time
import multiprocessing
import sys
from tqdm import tqdm

# Typical
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import datasets

# Classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Matplotlib
if not 'ipykernel' in sys.modules:
    # Prevents errors when plotting outside Jupyter Notebook
    # For more information see exercise 5
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt


# ## Funktions-Definitionen

# Um bei der Gittersuche einfach die Anzahl der CPU-Kerne variieren zu können, wird zunächst eine Funktion definiert, welche die Gittersuche (`GridSearchCV`) in einer `for`-Schleife mit einer jeweils anderen Anzahl an CPU-Kernen ausführt.

# In[2]:


def grid_search_with_cores(classifier, param_grid, X_train, y_train, core_range=None):
    """Instanciates `sklearn.model_selection.GridSearchCV` with given classifier/param_grid 
    and fits it on the given training data. This is done multiple times with different number of CPU cores.
    The numbers for CPU cores can be given with the core_range parameter.
    
    During computation, a progress bar is shown.
    
    Returns
    -------
    times : pandas.DataFrame
            Contains the time needed to train with the different numbers of CPU cores.
    grid_search: sklearn.model_selection.GridSearchCV
            Reference to one of the fitted grid search models.
    """
    
    # If no core range is given, take all possible values
    if core_range is None:
        core_range = list(range(1, multiprocessing.cpu_count()+1))
        
    # Instanciate DataFrame for saving the training times
    times = pd.DataFrame(columns=['n_cores', 'times'])

    # Show progress bar with tqdm
    for n in tqdm(core_range):
        # Instanciate Grid Search
        grid_search = GridSearchCV(classifier, param_grid, n_jobs=n)

        # Start time measurement
        start = time.time()

        # Modell trainieren
        grid_search.fit(X_train, y_train)

        # End time measurement
        end = time.time()

        # Append results
        times = times.append(pd.DataFrame.from_dict({'n_cores': [n], 'times': [end - start]}))

    # Return times Dataframe and reference to the last GridSearchCV object
    return times, grid_search


# Als nächstes definieren wir eine Funktion zum Plotten der Kurve mit der Suchzeit in Abhängigkeit von der Anzahl an CPU- Kernen:

# In[3]:


def plot_times(times):
    """Plots the times returned by grid_search_with_cores"""
    # Create the plot
    plot = times.plot(x='n_cores', y='times', legend=False)
    # Set plot labels
    plot.set(xlabel="#Cores", ylabel="Time in seconds")


# Zum Schluss definieren wir noch eine Funktion zum Plotten der Zielmetrik in Abhängigkeit von einem (bzw. zwei) ausgewählten Hyperparametern.

# In[4]:


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    """ Plots the mean test score from the information of `GridSearchCV.cv_results_` for two given grid parameters.
    
    Source: https://stackoverflow.com/a/43645640/6853900"""
    
    # Get Test Scores Mean for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    # Beautify the Plot (title, labels, legend, grid)
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


# ## Iris Flower Dataset

# ###### Laden und Vorbereiten des Datensatzes

# Siehe [Aufgabe 1](./Exercise1.ipynb) für mehr Details.

# In[40]:


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pd.read_csv(url, names=names)


# In[41]:


iris.shape


# In[42]:


# Split-out validation dataset like in the SDIL-Tutorial for comparability
array = iris.values
X = array[:,0:4]
Y = array[:,4]
test_size = 0.20
seed = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


# ###### Ausführen der Gittersuche

# Als Optimierungsparameter wurden die Parameter `C` und `kernel` gewählt.

# In[45]:


# Hyper-Parameter Space
grid_iris = {'C':[1,5,10, 50], 'kernel':['linear','rbf']}

# SVM als Klassifikator
svc = svm.SVC(random_state=0)

times_iris, search_iris = grid_search_with_cores(svc, grid_iris, X_train, y_train, core_range=range(1,33))


# ###### Plot der Suchzeit in Abhängigkeit von der Anzahl an CPU- Kernen

# In[47]:


print(times_iris)


# In[46]:


plot_times(times_iris)


# ###### Plot der Zielmetrik in Abhängigkeit vom Hyperparameter `C`

# In[48]:


plot_grid_search(search_iris.cv_results_, grid_iris['C'], grid_iris['kernel'], 'C', 'kernel')


# ###### Validierung auf den Test-Daten

# Zum Schluss soll die Performance des von der Gittersuche als besten identifizierten Klassifikators auf den Test-Daten ermittelt werden.

# In[17]:


# Print the best estimator
search_iris.best_estimator_


# In[18]:


# Predict with the best estimator and print the results
predictions = search_iris.best_estimator_.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# ## Heart Disease UCI Dataset

# Dieser Datensatz wurde auf [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci) gefunden.
# 
# Er enthält von 303 Patienten jeweils 14 Attribute und die Information darüber, ob sie eine Herzkrankheit haben oder nicht.
# 
# Die ursprüngliche Quelle für den Datensatz ist das [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

# ###### Laden und Vorbereiten des Datensatzes

# Die folgenden Zellen zum Laden und VOrverarbeiten der Daten sind zum Teil aus dem Kaggle Notebook "[What Causes Heart Disease?](https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model)" entnommen.

# In[19]:


heart = pd.read_csv("./data/heart.csv")


# In[20]:


heart.shape


# In[21]:


heart.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
                 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina',
                 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[22]:


heart['sex'] = heart['sex'].astype('object')
heart['chest_pain_type'] = heart['chest_pain_type'].astype('object')
heart['fasting_blood_sugar'] = heart['fasting_blood_sugar'].astype('object')
heart['rest_ecg'] = heart['rest_ecg'].astype('object')
heart['exercise_induced_angina'] = heart['exercise_induced_angina'].astype('object')
heart['st_slope'] = heart['st_slope'].astype('object')
heart['thalassemia'] = heart['thalassemia'].astype('object')


# In[23]:


heart = pd.get_dummies(heart, drop_first=True)


# In[24]:


heart.head()


# In[25]:


#split the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(heart.drop('target', 1), heart['target'], test_size = .2, random_state=0) 


# ###### Ausführen der Gittersuche

# Als Optimierungsparameter wurden die Parameter `n_estimators` und `max_features` gewählt.

# In[30]:


# Hyper-Parameter Space
grid_heart = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)],
              'max_features': ['auto', 'sqrt']}

# SVM als Klassifikator
clf = RandomForestClassifier(random_state=0)

times_heart, search_heart = grid_search_with_cores(clf, grid_heart, X_train=X_train, y_train=y_train, core_range=range(1,33))


# ###### Plot der Suchzeit in Abhängigkeit von der Anzahl an CPU- Kernen

# In[31]:


print(times_heart)


# In[32]:


plot_times(times_heart)


# ###### Plot der Zielmetrik in Abhängigkeit vom Hyperparameter `n_estimators`

# In[33]:


plot_grid_search(search_heart.cv_results_, grid_heart['n_estimators'], grid_heart['max_features'], 'n_estimators', 'max_features')


# ###### Validierung auf den Test-Daten

# Zum Schluss soll die Performance des von der Gittersuche als besten identifizierten Klassifikators auf den Test-Daten ermittelt werden.

# In[34]:


# Print the best estimator
search_heart.best_estimator_


# In[35]:


# Predict with the best estimator and print the results
predictions = search_heart.best_estimator_.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

