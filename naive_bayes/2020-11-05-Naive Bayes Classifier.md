---
layout: post
title: Naive Bayes Classifer
key: 20201105
tags:
  - Python
  - Pytorch
  - Naive Bayes
  - machine learning
---

# Naive Bayes Classifiers

## Definition

Naive Bayes classifiers are based on naive bayes classifier algorithms. Let's say we have m input values <img src="https://latex.codecogs.com/svg.latex?  \overrightarrow{x} =< x_{1},x_{2},x_{3},...,x_{m} >" title="x_{ij}" />

1. Assume all these input variables/features are conditionally independent given Y. 
In reality it is not quite `possible` that all features are conditionally independent
{:.info}

2. Simply chose the class label that is the most likely given the data, predict Y using <img src="https://latex.codecogs.com/svg.latex?   \widehat{Y} =  \argmax_{y}  P( \overrightarrow{x},Y)" title="x_{ij}" />




## String and Math Operation
if when looping through string/array, current variable has direct influence to previous variable and both variable could be removed or modified after operation, then considering using stack as data structure.

* 227 Basic Calculator II([Q](https://leetcode.com/problems/basic-calculator-ii/):[A]())(stack)

keep in mind data structure:
stack::top() -> sign -> tmp -> s[i]  
{:.info}
