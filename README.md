## Novel Author Classification
An accurate authorship identification system can be critical to solving authorship dispute problems in texts. For example, this is especially useful for detecting plagiarism detection and labeling ancient documents as the amount of digital literature grows rapidly. In this project, we look at the effectiveness of techniques for feature selection and classification models in the context of discriminating between 11 authors.

## The Data
The data consists of 54 classical full text novels collected from the Gutenberg project. To imitate the context of classifying a single page of a novel, we divide the raw text into samples by 2,500 characters (including white-spaces), the average in a page of novel, and ending at the nearest word. Though the distribution of the authors’ samples is highly skewed, but this gives us a more realistic insight into how our models perform when the authors have a varying number of past works that we can use as training samples.

## Feature Engineering
My system attempts to capture useful words and phrases by extracting the frequency-inverse document frequency (TFIDF) of uni-grams and bi-grams having 0.001 to 0.1 document frequency to filter out words and phrases that are too rare or common. The vectorizer also ignores common English stop-words. Next, we perform a feature ranking to see the most informative features through an ANOVA test. The mean frequency of each feature between the 11 classes is compared through an F statistic that indicates whether the feature is informative in classification

Currently, our data is represented in a 8,829x23,379 sparse matrix. Since many phrases and words are used in the same contexts, we can use Principal Component Analysis (PCA) to reduce the dimension of the data by compacting the correlating features in eigenvectors. By capturing 90% of the original variance, we reduce the number of features from 23,379 to 3,953 while sacrificing the interpret-ability of the final results. 

Now that we have reduced the columns drastically by compacting the variances of correlation features, we can further filter out ones that don’t correlate with the author labels by selecting the top 1,000 columns of the PCA results that have the highest F-test statistics. Overall, the number of features has been reduced from 23,379 to 1,000, while maintaining most of the information.

Although cross-validation would yield a slightly more realistic picture of how our models perform on unseen data, training each model multiple times on a matrix of this size on a layman’s laptop would only drive us closer to insanity. Therefore, we randomly split the data into 75% for training and 25% for testing.

## Performances
As a basis for comparison, a naive random classifier has an expected accuracy of 1/11 = 0.091. I train six types of classifiers on the training data, and evaluate them on the testing set through four metrics: accuracy, precision, recall, and f-score (macro averaged). To perform hyper-parameter tuning on SVM, Random Forest, and AdaBoost, we train four models of each with varying parameters and select the one that yields the highest accuracy.

Neural network architecture:

1.  1,000 input neurons, ReLU, 25% dropout
    
2.  100 densely-connected neurons, ReLU, 25% dropout
    
3.  11 output neurons, Softmax

## Conclusions
Overall, the Neural Network and linear SVM classifiers yield the highest accuracies and F-scores. This indicates a linear decision boundary in the data and that predicting authors using the features extracted from our process is informative. To further show the process’s effectiveness, we train and test a linear SVM model on the raw features before the PCA and filtering with the same model parameters.

*Linear SVM (C=1.8) on raw features.*
Accuracy, Precision, Recall, F-score
0.952       0.945       0.891    0.914

Despite the slightly improved accuracy (0.949 to 0.952), the F-score decreased (0.927 to 0.914), indicating that there’s no loss of significant variance from the vectorizer’s outputs.

In summary, the neural network’s 95.4% classification accuracy fulfills the goal of discriminating between a small number of authors with varying amounts of available data. Whether this system can withstand a larger database with hundreds or thousands of authors is an intriguing question moving forward. Perhaps new features need to be extracted and online training models used so that new information can be added iteratively.
