from naivebayes import *

# create model
nb = NaiveBayes.NaiveBayes()

# train
nb.fit(train_df, train_labels)

# predict
output = nb.predict(test_df)
