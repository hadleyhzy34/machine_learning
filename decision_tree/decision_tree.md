## import modules


```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
```

## training data in our test example


```python
training_data=[
    ['Green',3,'Apple'],
    ['Yellow',3,'Apple'],
    ['Red',1,'Grape'],
    ['Red',1,'Grape'],
    ['Yellow',3,'Lemon'],
]
print(training data)
```

## question class


```python
class Qeustion:
    # a question is used to partition a dataset
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
    def match()
```


```python
## demo
# let's write a question for a numeric question
Question(1,3)
# let's write a question for a categorical attribute
Question(0,'Green')
```

## partition function


```python
def partition(rows, question):
    #partitions dataset, rows: dataset question is used to seperate dataset into two different list
    true_rows, false_rows = [],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
```

demo to show how to partition the training data based on whether rows are red:


```python
true_rows, false_rows = partition(training_data, Question(0,'Red'))
true_rows
print(1)
```

## Gini Impurity

## Information Gains


```python
def find_best_split():
```


```python
## build the tree using recursion
```


```python

```
