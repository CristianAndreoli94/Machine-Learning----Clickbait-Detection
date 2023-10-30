# Machine-Learning----Clickbait-Detection
Small university project about Machine Learning

## Problem definition and data
Clickbait headlines are designed to attract attention and to entice users to read the linked piece of online content. They are typically deceptive, sensationalized, or otherwise misleading. More importantly, they do not accurately reflect the content being delivered.
A software house wants to build a browser capable of distinguishing clickbaits from regular headlines. To do so a dataset has been collected, including 32 000 headlines, equally divided in the ‘clickbait’ and ‘non-clickbait’ classes. The dataset includes a training, validation, and test sets consisting of 24 000, 4000 and 4000 samples, resepctively. The data is stored in text files, with one headline for each line.

## Assignment
We want to build a classifier that is able to predict if a given headline text is actually a clickbait. For the programming assignment you are expected to:
- analyze and comment the data;
- design and implement a suitable data pre-processing procedure;
- implement, train and evaluate one or more classification models;
- use suitable data processing and visualization techniques to analyze the behavior of the trained models.
Two scenarios should be considered: a generic one, in which all errors are equally important, and a ‘precision-oriented’ scenario, in which we would like to keep as small as possible che chance of false positives.
All the above should be implemented as scripts in the Python programming language.
Any machine learning library (included pvml) can be used.
