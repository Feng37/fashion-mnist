# fashion-mnist
For problem 1, I tried some basic models first. Although it takes short time for learning , the accuracy is not that high. For example, logistic regression takes less than 10 min with accuracy 0.8133. KNN takes 44 min with accuracy 0.8446. I tried these simple models because sometimes simple model would give better result tham those complex model. Then I tried CNN and LSTM. I tried LSTM because some papers say that LSTM is also suitable for some image processing tasks. However, in this problem LSTM only gets accuracy 0.8105 in 20 min. CNN gets the best result in around 300 min with accuracy 0.9205. If we use batch size 64 instead of 128, we would only use around 140 min to learn, but the accuracy is 0.9163. 



The solution to problem 2 is in floder ''test-eigen''. This is a classic Reservior Sampling problem. The code is in C++. 
