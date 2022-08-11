# **Machine Learning Projects**
## [Sentiment analysis (NLP)](https://github.com/berndtmihaly/data-science-projects/blob/main/Covid%20sentiment%20analysis.ipynb)
- The dataset contained tweets about COVID labelled by sentiments. My objective was to build an NLP pipeline and train a neural network to classify the tweets both at a fairly deep level of abstraction.
- My aim was to experiment with pre-trained embeddings, continue train them with transfer learning and try different regularization techniques.
- The high level structure of my approach was: data cleaning, tokenization, load pre-trained embedding, train the network.
- I have achieved a fairly similar balanced accuracy with and without the pre-trained embeddings. Data preparation was important, the usage of packed padded sequences significantly reduced training time and the results of regularizations were noticeable.
- Proposal for further analysis: train own word embeddings on a corpus with more information about COVID, classification with BERT, more efforts to clean the data.

<p align="center">
  <img src="/images/pca.png" alt="PCA" align="center" style="width: 550px;">
</p>
<p align = "center">
  <i>First two principal components of GloVe word embeddings before training</i>
</p>
<br>

<p align="center">
  <img src="/images/early%20stopping.png" alt="Learning curves" align="center">
</p>
<p align = "center">
<i>Loss function and balanced accuracy with early stopping</i>
</p>

## [Classification of medical data with SVM](https://github.com/berndtmihaly/data-science-projects/blob/main/Berndt_Mih%C3%A1ly_SVM_Classification.ipynb)
- The dataset contained tabular medical data with numerous missing data. The problem was introduced in the Data Mining Models and Algorithms course, and the objective was to classify the cases using SVM. I finished in second place.
- The high level structure of my approach was: exploratory data analysis, outlier detection, feature engineering (encoding, transformations, scaling, feature selection), missing data imputation with KNN and apply SVM.
- I used the following embedded method for feature selection. I trained a Random Forest and then used permutation importance to select features. I then performed the classification using SVM, running experiments with all possible Kernels. For hyperparameter optimization I used a Bayesian approach with Optuna.
- Proposal for further analysis: the dataset is augmented with the chest X-ray images of the patients. A possible approach could be to train a neural network to classify the images, then extract the features and add them to the tabular observations to perform the classification using SVM.

Confusion matrix on the validation set             |  Classification report on the validation set
:-------------------------:|:-------------------------:
![](/images/svm%20cm.png)  |  ![](/images/svm%20class%20report.png)

## [Optimize Boosting algorithms for image classification](https://colab.research.google.com/drive/1b0i2a5Hxji9hWAwDTzxXV2VhNamEiRYs?usp=sharing)
![](https://github.com/berndtmihaly/data-science-projects/blob/main/images/xgboost.JPG)
![](/images/xgboost2.JPG)
