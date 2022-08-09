# **Machine Learning Projects**
## **Sentiment analysis (NLP)**
- The dataset contained tweets about COVID labelled by sentiments. My objective was to build an NLP pipeline and train a neural network to classify the tweets both at a fairly deep level of abstraction.
- My aim was to experiment with pre-trained embeddings, continue train them with transfer learning and try different regularization techniques.
- The high level structure of my approach was: data cleaning, tokenization, load pre-trained embedding, train the network.
- I have achieved a fairly similar balanced accuracy with and without the pre-trained embeddings. Data preparation was important and the results of the regularisations were noticeable.
- Further analysis: train own word embeddings on a corpus with more information about COVID, classification with BERT, more efforts to clean the data.

| ![](./images/pca.png =250x) | 
|:--:| 
| *Visualization of pre-trained GloVe word embedding with PCA* |

| ![](/images/learning%20curve.png) | 
|:--:| 
| *Loss function and balanced accuracy with early stopping* |

## [Imbalanced classification SVM](https://github.com/berndtmihaly/data-science-projects/blob/main/Berndt_Mih%C3%A1ly_SVM_Classification.ipynb)
![](https://github.com/berndtmihaly/data-science-projects/blob/main/images/svm%20cm.png)
![](https://github.com/berndtmihaly/data-science-projects/blob/main/images/svm%20class%20report.png)

## [Optimize Boosting algorithms for image classification](https://colab.research.google.com/drive/1b0i2a5Hxji9hWAwDTzxXV2VhNamEiRYs?usp=sharing)
![](https://github.com/berndtmihaly/data-science-projects/blob/main/images/xgboost.JPG)
![](/images/xgboost2.JPG)
