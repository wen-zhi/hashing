Python implementations of hashing model
---------------------------------------

The implemented hashing models include:
- LSH
- PCA-ITQ, CCA-ITQ
- SDH

Requirements
------------

```
# create a new env
conda create --name hashing scikit-learn numpy
# activate the new env
conda activate hashing
```

Usage
------

The current implemented models are `LSH, ITQ, ITQ_CCA, SDH`. All the model share the same interface,
so it is very easy to use them. The Following are two examples:

> You can download **MNIST-GIST512D** from [here](https://drive.google.com/open?id=14MG9OGekFROlbe-aLHuNlRMiFZITHFhh).

## ITQ

```python
from hashing.model import ITQ
from hashing.dataset import load_mnist_gist
from hashing.evaluation import mean_average_precision, precision_recall

# load mnist-512d data
root = './datasets/mnist-gist-512d.npz'
query_data, train_data, database_data = load_mnist_gist(root)
query, query_label = query_data
train, _ = train_data
database, database_label = database_data

# ITQ
encode_len = 32
model = ITQ(encode_len)
model.fit(database)

# encode
query_b = model.encode(query)
database_b = model.encode(database)

# evaluation
mAP = mean_average_precision(query_b, query_label,
                             database_b, database_label)
precision, recall = precision_recall(query_b, query_label,
                                     database_b, database_label)
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)
```

## ITQ-CCA

```python
from hashing.model import ITQ_CCA
from hashing.dataset import load_mnist_gist
from hashing.evaluation import mean_average_precision, precision_recall
from hashing.utils import one_hot_encoding

# load mnist-512d data
root = './datasets/mnist-gist-512d.npz'
query_data, train_data, database_data = load_mnist_gist(root)
query, query_label = query_data
train, train_label = train_data
database, database_label = database_data

# ITQ
model = ITQ_CCA(encode_len=32)
train_label = one_hot_encoding(train_label, 10)
model.fit(train, train_label)

# encode
query_b = model.encode(query)
database_b = model.encode(database)

# evaluation
mAP = mean_average_precision(query_b, query_label,
                             database_b, database_label)
precision, recall = precision_recall(query_b, query_label,
                                     database_b, database_label)
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)
```

