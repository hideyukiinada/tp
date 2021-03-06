# Accuracy chart

Dataset: 20 Newsgroups
http://qwone.com/~jason/20Newsgroups/

20news-18828.tar.gz (duplicates removed, only "From" and "Subject" headers)

## Size of dataset

Type|Size|
|---|---|
|Training dataset| 15069|
|Test dataset| 3759|
|Total | 18828|

This particular dataset (20news-18828.tar.gz) was not split into training and test dataset, so
it was split into two sets for these testing by me.

## Discrepany in accuracy between machines to run test
I observed that accuracy tends to be better on Linux than Mac for an exact same test script.  I haven't had a chance to look into this further.  Therefore if you see different numbers in below tables, that means that they were run on different machines.

Type|Vocabulary|Epoch|Loss| Training Accuracy | Test Accuracy |
|---|---|---|---|---|---|
|TF-IDF + Neural Network | NA | 5 | 0.0103 | 0.9985 | 89.917531% |
 TF-IDF + Neural Network | NA | 1 | 1.4383 | 0.7987 | 87.975525% |
|TF-IDF + Naive Bayes | NA | NA | NA | NA | 87.682894% |
|TF-IDF + SVM | NA | NA | NA | NA | 84.916201% |
|TF (no IDF) + Neural Network|1000|20|0.0731 | 0.9906 | 61.479117%
|TF (no IDF) + Neural Network|10000|20|0.0380 | 0.9946 | 79.675446%
|TF (no IDF) + Neural Network (2nd run)|10000|20|0.0256 | 0.9981| 81.723863%
|Embedding with pooling + Neural Network(256 words, 16 embedding output, truncating:pre) | 10000 | 20 | 0.2864 | 0.9377 | 71.827614% |
|Embedding with pooling + Neural Network(256 words, 16 embedding output, truncating:post) | 10000 | 20 | 0.2833 | 0.9385 | 73.370577% |
|Embedding with pooling + Neural Network(256 words, 128 embedding output, truncating:post) | 10000 | 20 | 0.0123 | 0.9982 | 77.866454% |
|Embedding with pooling + Neural Network(512 words, 128 embedding output, truncating:post) | 10000 | 20 | 0.0392 | 0.9932 | 76.509710% |
|Embedding without pooling + Neural Network(512 words, 128 embedding output, truncating:post) | 10000 | 20 |  0.0457 | 0.9869 | 66.374036% |
|Embedding + LSTM | 10000 | 20 | 0.2826 | 0.9207 | 71.481777%|
|Embedding word2vec + Neural Network (300D, all words) | All | 10,000 | 0.8653 |0.7168 | 43.522213% |
|Embedding doc2vec + Neural Network (100D, no word normalization) | All | 20 | 0.1089 | 0.9696 | 60.148976% |
|Embedding doc2vec + Neural Network (50D, no word normalization) | All | 20 | 0.1720 | 0.9472 |58.286778% |
|Embedding doc2vec + SVM (50D, no word normalization) | All | 20 SVM | NA | NA |52.540569% |
|Embedding doc2vec + Neural Network(50D, no word normalization, dropout added) | All | 20 | 0.9443 | 0.6763 |60.681032% |
|Embedding doc2vec + Neural Network (50D, no word normalization, dropout added) | All | 40 | 0.8516 | 0.7024 |60.707635% |
 Embedding doc2vec + Neural Network (100D, no stop words, bigram support) | All | 20 | 0.1265 | 0.9668 | 40.622506% |

## Epoch and accuracy for TF-IDF Neural Network approach

Measured on Linux

Type|Vocabulary|Epoch|Loss| Training Accuracy | Test Accuracy |
|---|---|---|---|---|---|
|TF-IDF + Neural Network | NA | 1 |1.4305  | 0.8095 | 91.726523% |
|TF-IDF + Neural Network | NA | 5 |0.0112   | 0.9983 | 93.615323% |
|TF-IDF + Neural Network | NA | 10 | 0.0057 | 0.9987 | 93.402501% |
|TF-IDF + Neural Network | NA | 20 | 0.0030 | 0.9991 | 93.482309% |

## Machines used
* Mac (OS:10.13.5, RAM: 16 GB, CPU: 2.6 GHz Intel Core i5, Python: 3.6.7) 
* Linux (OS: Ubuntu 18.10, RAM: 48 GB, CPU: 3.4 GHz Intel Core i5-7500, GPU: NVIDIA GeForce GTX 1080, Python: 3.6.7) 
