Accuracy history

Size of dataset

Type|Size|
|---|---|
|Training dataset| 15069|
|Test dataset| 3759|


Type|Vocabulary|Epoch|Loss| Accuracy | Test Accuracy |
|---|---|---|---|---|---|
|Naive Bayes | NA | NA | NA | NA | 87.682894% |
|SVM | NA | NA | NA | NA | 84.916201% |
|Word frequency matrix per doc + Neural Network|1000|20|0.0731 | 0.9906 | 61.479117%
|Word frequency matrix per doc + Neural Network|10000|20|0.0380 | 0.9946 | 79.675446%
|Word frequency matrix per doc + Neural Network (2nd run)|10000|20|0.0256 | 0.9981| 81.723863%
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
