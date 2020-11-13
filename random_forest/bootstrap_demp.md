```python
import numpy as np
import seaborn as sns
import random
```

normal distribution dataset


```python
x = np.random.normal(loc=500.0,scale=1.0,size=10000)
np.mean(x)
```




    500.0033021822781




```python
sample_mean = []

#bootstrap sampling
for i in range(10000):
    y = random.sample(x.tolist(), 5)
    avg = np.mean(y)
    sample_mean.append(avg)
    # print(avg)
```


```python
np.mean(sample_mean)
```




    500.00585544721156




```python

```
