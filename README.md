# Ai_sms_spam_detector

I will assume you already eat python for breakfast to install yourself required libraries to run this project, In case any trouble feel free to leave questions!

[+] K Nearest Neighbors Accuracy: 92.6776740847%\n
[+] Decision Tree Accuracy: 97.4874371859%\n
[+] Random Forest Accuracy: 97.7027997128%\n
[+] Logistic Regression Accuracy: 98.8513998564%\n
[+] SGD Classifier Accuracy: 98.2770997846%\n
[+] Naive Bayes Accuracy: 98.6360373295%\n
[+] SVM Linear Accuracy: 99.0667623833%\n


                              
               precision    recall  f1-score   support

           0       0.99      1.00      0.99      1208
           1       0.98      0.95      0.96       185

   micro avg       0.99      0.99      0.99      1393
   macro avg       0.99      0.97      0.98      1393
weighted avg       0.99      0.99      0.99      1393

             predicted
             ham spam

actual	ham	1205	  3
       spam	  10	175
     
|              |               |                |   Predicted   |
| :---         |     :---:     |     :---:      |          ---: |
|              |               |      HAM       |     SPAM      |
|    ACTUAL    |      HAM      |     1205       |       3       |
|              |     SPAM      |       10       |     175       |                             
