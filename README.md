# Anomaly Detection

## <b>Content </b>
 * [How to run it](#how-to-run-it)
 * [Introduction and Motivation](#introduction-and-motivation)
 * [Libraries Used](#libraries-used)
 * [Project Report](#Project-Report)
 * [Additional Resources](#additional-Resources)


## How to run it

 * Pull the src code
 * Open terminal
 * Navigate to the folder
 * Run
```Bash
    make docker_build
    make docker_run
```
 * For results: AnomalyDetectionProject > data > .png files


## Introduction and Motivation
This repo was created as project for CPSC-473: Introduction to Data Mining course. 
For the course my classmate @TheLostLight and I created a simple anomaly detection python project.
Out of vast variety of anomaly detection techniques available we picked Guassian Distribution. The reason why we selected Guassian Distribution is because it is a very popular, easy to explain and implement. If you are interested in more about this check out the project report below.


## Libraries Used
For normal user program:
 * python3
 * pandas
 * numpy 
 * matplotlib
 * openpyxl

For dev (including above libraries):
 * isort
 * black
 * flake8
 * safety
 * pdoc3


## Project Report

[Link to project report](https://www.overleaf.com/read/zbbtsjkpwjyz)


## Additional Resources
 * [Precision and Recall](https://towardsdatascience.com/a-complete-understanding-of-precision-recall-and-f-score-concepts-23dc44defef6)
 * [Intro to Anomaly Detection](https://towardsdatascience.com/introduction-to-anomaly-detection-c651f38ccc32)
 * [Step by step guide: Anomaly detection Algorithm](https://towardsdatascience.com/a-complete-anomaly-detection-algorithm-from-scratch-in-python-step-by-step-guide-e1daf870336e)
 * [Cross Validation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiTmOzFh8PtAhUWuZ4KHRioAcMQFjADegQICBAC&url=https%3A%2F%2Fmachinelearningmastery.com%2Fk-fold-cross-validation%2F&usg=AOvVaw3dq5uCuxWnEEzbZGBP5vK2)
