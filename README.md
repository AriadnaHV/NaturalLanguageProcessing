The goal of this project was to perform sentiment analysis on approx. 300k reviews on sports and outdoors items. A variety of Natural Language Processing (NLP) techniques were used in the overall pipeline, which was divided into the following four parts:

1) Choose a dataset, download and decompress it, load it into a Python dictionary, and proceed to explore the data contained in it.
   
2) Informed by the data exploration process, design and implement a full preprocessing pipeline.
   
3) Train two models with different parameters, compare them using appropriate metrics, and select one as the final model for binary supervised classification. Run it on the test set.

4) Calculate appropriate metrics to evaluate the performance of the model. Include final comments and conclusions.

Files:

* AHV_01_NLP.ipynb -> Part 1) Data exploration
* AHV_O2_NLP.ipynb -> Parts 2) Preprocessing, 3) Training and Testing, 4) Evaluation and conclusions
* lr_clf.pkl -> Trained Logistic Regression Classifier model from Part 3)
* rf_clf.pkl -> Trained Random Forest Classifier model from Part 3)
* model_eval_results.pkl -> Test results of models
* requirements.txt -> list of external libraries needed for AHV_01_NLP.ipynb
* requirements2.txt -> list of external libraries needed for AHV_02_NLP.ipynb
* utils.py -> helper functions for AHV_01_NLP.ipynb
* utils2.py -> full preprocessing pipeline and helper functions for AHV_O2_NLP.ipynb
* w2v_model.model -> Word2Vec model used in Part 1)
