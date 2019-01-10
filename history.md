Accuracy history

Vocabulary size:  188459

Type|Vocabulary|Epoch|Loss| Accuracy | Test Accuracy |
|---|---|---|---|---|---|
|Word frequency matrix per doc|1000|20|0.0731 | 0.9906 | 61.479117%
|Word frequency matrix per doc|10000|20|0.0380 | 0.9946 | 79.675446%
|Word frequency matrix per doc (2nd run)|10000|20|0.0256 | 0.9981| 81.723863%
|Embedding with pooling (256 words, 16 embedding output, truncating:pre) | 10000 | 20 | 0.2864 | 0.9377 | 71.827614% |
|Embedding with pooling (256 words, 16 embedding output, truncating:post) | 10000 | 20 | 0.2833 | 0.9385 | 73.370577% |
|Embedding with pooling (256 words, 128 embedding output, truncating:post) | 10000 | 20 | 0.0123 | 0.9982 | 77.866454% |
|Embedding with pooling (512 words, 128 embedding output, truncating:post) | 10000 | 20 | 0.0392 | 0.9932 | 76.509710% |
|Embedding without pooling (512 words, 128 embedding output, truncating:post) | 10000 | 20 |  0.0457 | 0.9869 | 66.374036% |
|LSTM with embedding | 10000 | 20 | 0.2826 | 0.9207 | 71.481777%|
|Naive Bayes | NA | NA | NA | NA | 87.682894% |

