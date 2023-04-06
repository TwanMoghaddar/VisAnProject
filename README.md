"""
@Author: Antoine Moghaddar, Casper Dert, Ibrahim Ahmed
@Date: 30/03/2023
@University of Eindhoven; MSc Data Science & AI
@Group 18

@Contents

For the overall implementation of the project we have made use of the Dash toolbox/framework.
By using libraries such as dash_bootstrap_components and Dash.html, we were able to host a webpage with our analysis
results and visualizations.

Additionally, we have done some on-top data analysis by making use of various techniques. For the sentiment analysis we
made use of NLPs (Natural Language Processing.)
This code is a collection of functions that perform natural language processing (NLP) on a given CSV dataset.
NLP Algorithm using the VaderSentiment and Textblob library to classify tweets into categories of positivity.
The classification is built upon the principle of text splitting and language processing wherein a blacklist of
words is used to determine the tone and the mental intentions of the tweet.

Moreover, for the similarity check we made use of the deep learning. This code calculates the similarity between
individuals by creating a matrix where each column represents a different feature of the analysis and each row
represents a different person. The first n columns represent the frequency of a specific item in all the tweets of each
person using the CNN Exception model. The last three features are the averaged Stiffness, Polarity, and Tone of all
tweets for each person. Stiffness is defined as the ratio of "emotionless/harsh" words to "emotionful" words, Polarity
is the overall sentiment of the tweets, and Tone refers to the choice of words and the number of possible
interpretations. The distance between persons is calculated using the pairwise_distances function from the sci-kit
library. The function takes two arguments: predictions_df, which is a Pandas Dataframe that contains picture id,
classification, percentage of certainty, and rank of said classification, and text_analysis_dictionary, which is a
dictionary where the key is "Personx" and the value is a list of values for Stiffness, Polarity, and Tone. The function
returns a numpy matrix of size p times p, where p is the number of individuals in the dataset. The values in the matrix
represent the distance between two persons, row and column respectively.

For the co-dependency of users and persons, we made use of a system as described below.
The provided system visualizes images in conjunction with their top five predicted classes and associated probabilities.
The predictions are generated using a generic image classification model, Xception, which has not been specifically
trained on the dataset under consideration. Xception is a 71-layer deep convolutional neural network that is pre-trained
on a vast dataset of over a million images from the ImageNet database, enabling it to classify images into 1000 object
categories. The interactive user interface of the system facilitates exploration of predictions for each image with
ease. Users can simply click on an image cell within the table to display the selected image alongside a detailed
prediction table located below it. This table lists the predicted classes and their respective probabilities, thereby
enabling users to delve deeper into the model's classification decisions.

For more information, please check our report.