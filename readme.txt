Cornell movie review sentence polarity data can be found here: http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz

The Matlab NLP toolbox used for preprocessing can be found here: https://github.com/faridani/MatlabNLP

The code in MRsentiment.py was based on the script found here, although some significant changes were made:
https://github.com/emanuele/kaggle_pbr/blob/master/blend.py

The data was first imported into Matlab for preprocessing, then written to csv files. The csv files were read and classification models built in python.