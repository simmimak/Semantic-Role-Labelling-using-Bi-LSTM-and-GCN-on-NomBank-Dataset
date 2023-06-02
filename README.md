# Nombank Argument Prediction

This project aims to predict arguments given a predicate using the Nombank dataset. It employs a combination of Bi-LSTMs and Graph Convolutional Networks (GCNs) to achieve accurate predictions.

## Dataset

The Nombank dataset was used for training and evaluation. The training dataset consisted of 2174 sentences, while the test dataset had 150 sentences. The data was preprocessed by encoding the token, POS (Part-of-Speech), and BIO (Begin, Inside, Outside) features into numeric values.

## Preprocessing

The preprocessing phase involved merging the tokens based on the sentence ID to form sequences suitable for the Bi-LSTM implementation. The same preprocessing was applied to the POS and BIO tags. Additionally, dependency head features were extracted using a Spacy Model, which were mainly used in the GCN model.

## Models

Several models were implemented for comparison, including Adaboost, Random Forest, and Logistic Classifier, which utilized various features such as token distances, previous/next tokens, POS tags, and BIO tags. These models served as baselines for evaluating the proposed approach.

The primary model used in this project was the Bi-LSTM combined with GCNs. Initially, an LSTM layer was stacked on top of the Bi-LSTM layer, achieving a F1-score of 0.80. To improve recall, the stacked LSTM layer was removed, resulting in a higher F1-score of 0.81, albeit with a decrease in precision.

The modified GCN + Bi-LSTM architecture did not show significant improvements compared to the Bi-LSTM model, but it outperformed the initial model with an LSTM layer. Further enhancement of the GCN implementation, such as adding more GCN layers, may yield better results by capturing dependency information.

A softmax classifier was also employed, which showed significant improvement over the previous models. The softmax with LSTM achieved a precision of 0.88 and an F1-score of 0.82, while the softmax without LSTM achieved the best F1-score of 0.84.

## Conclusion

Based on the experimental results, the combination of a Bi-LSTM model without an LSTM layer and a softmax classifier achieved the highest performance on the Nombank dataset's percentage class, with an F1-score of 0.84. Further hyperparameter tuning may lead to further improvements. 

## Code

The code of this project is shared in ArgumentPrediction.ipynb notebook.

