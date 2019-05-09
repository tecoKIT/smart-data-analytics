# Übung 1

Dieses Projekt enthält die Lösungen für das [erste Übungsblatt](./Uebungsblatt_1_Grundlagen.pdf). Das Ziel des Projekts ist die Wiederholung einiger Grundlagen der Statistik und, wie die Eigenschaften eines Datensatzes die Analyse beeinflussen.

Die Analysen sind ausschließlich in Python durchgeführt.

Installationsanweisungen / Abhängigkeiten

Dieses Projekt arbeitet an der Conda-Umgebung.

1. Installieren Sie zunächst die Anaconda Python-Distribution.
2. Neue Conda-Umgebung erstellen und  aktivieren 
3. Installieren Sie die benötigten Bibliotheken mit
          
        conda install --file requirements.txt
        
Inhaltsverzeichnis

1. Ordner `data`: Enthält die verfügbaren Datensätze
2. Ordner `module`
    - `beta_regression.py`: Statsmodels Beta Regression Wrapper 
    - `one_hot.py`: One-Hot-Encoding Hilfsmethoden
    - `poisson_regression.py`: Statsmodels Poisson Regression Wrapper
3. `Exercise*.ipynb`: Die Lösungen zu den jeweiligen Aufgaben
4. `requirements.txt`: Die Datei beinhaltet die verwendeten Python-Bibliotheken und ihre Versionsnummern.