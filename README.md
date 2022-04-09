# Assignment 1: Text Classification

In this programming assignment, we have to build a supervised text classifier i.e., sentiment classification with given dataset. The main goal is to classify the sentiment/label of user reviews into broadly two classes negative and positive reviews. Accuracy obtained on Base Model: With logistic regression and white space tokenizer:

Accuracy on train is: 0.9821038847664775 

Accuracy on dev is: 0.777292576419214

## Libraries required for for 2.1

```bash
import tarfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
```

## Libraries required for for 2.2

```bash
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import tarfile
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
```


## Running

For running: the code is in executable form in Jupyter Notebook, run all the cells to get desired output. Name of the files

```bash
2.1 Guided Feature Engineering.ipynb
2.2 Independent Feature Engineering.ipynb
````
