import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Question:
    # a question is used to partition a dataset
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match(self, example):
        # compare the feature value in an example to the feature
        # value in this quesiton
        val = example[self.column]
        return val >= self.value

def partition(rows, question):
    #partitions dataset, rows: dataset question is used to seperate dataset into two different list
    true_rows,false_rows=[],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    # """ calculate the gini impurity for a list of rows"""
    datasets = np.array(rows)
    impurity = 1
    if datasets.size == 0:
        return 0
    #number of labels
    unique,counts=np.unique(datasets[:,-1],return_counts=True)
    for label,count in zip(unique,counts):
        # print(label,count)
        prob = count / float(np.size(rows,0))
        impurity -= prob**2
    return impurity

def info_gain(left, right, current_gini_impurity):
    #the uncertainty of the starting node, minus the weighted impurity of two child nodes
    p = float(len(left)) / (len(left) + len(right))
    return current_gini_impurity - p * gini(left) - (1-p) * gini(right)


def find_best_split(rows):
    #find the best question to ask by iterating over every feature/value and calculating the information gain
    current_impurity  = gini(rows)
    best_gain = 0
    best_question = None
    n_features = len(rows[0])-1

    datasets = np.array(rows)
    for col in range(n_features):
        unique,counts=np.unique(datasets[:,col],return_counts=True)
        for val,count in zip(unique,counts):
            question = Question(col,val)
            true_rows,false_rows = partition(rows,question)
            gain = info_gain(true_rows, false_rows, current_impurity)
            # print(col, val, gain)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.datasets = np.array(rows)
        self.unique, self.counts = np.unique(self.datasets[:,-1], return_counts=True)
        max_count = 0
        for val, count in zip(self.unique, self.counts):
            if count > max_count:
                max_count = count
                self.predictor = val

class Decision_Node:
    def __init__(self,question,true_branch,false_branch):  
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node):
    if isinstance(node, Leaf):
        print("leaf")
        print((node.unique,node.counts))
        return
    
    print("question is: ")
    print(node.question.column, node.question.value)

    print ('--> True:')
    print_tree(node.true_branch)

    print ('--> false:')
    print_tree(node.false_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictor

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


