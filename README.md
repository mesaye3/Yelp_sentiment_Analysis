# Yelp_sentiment_Analysis

The following sentimental analysis was conducted on the yelp_review dataset. The goal of this analysis is to be able to read and analyze the review text data and provide positive or negative sentiment associated with the text. There are four stages to go through for a successful sentimental analysis!

# ![image](https://user-images.githubusercontent.com/97130281/148303597-8a5490bc-f1ab-419c-92cf-780465034250.png)


# Data preprocessing

Data preprocessing needs to be performed before placing the review text in the machine learning model. Data preprocessing was primarily done on the “text” dataset. The “stars” column contains a rating on a scale of 1 to 5. The rating must be standardized into binary values for the sentimental analysis and ratings above 4 is classified as 1 (positive review) and rating below 2 is classified as 0 (negative review). A new column called “sentimental rating” was created to capture this information.
NLTK (natural language toolkit) allows us to transform the raw review text containing irrelevant information into a structured review text. First, the string library was used to remove special characters (! @, #, $) from the raw review text, the removal of special characters enables the model to focus only on the text. Second, word tokenization was performed to extract each word, and this was achieved using word_tokenize from the nltk library and regular expression was used to split text on whitespace. Third, the machine learning model considers the same lowercase and uppercase form of a word, as two different words. Thus, tokenized text was used to convert the text into lowercase using the lower () function. Fourth, english stop words like (I, me, my) must be removed from the text using stopwords from nltk.corpus because it does not add significance to the text. Fifth, applying stemmer on the review text helps the model understand words better. Stemmer reduces words to their root forms, (running, runner changed to run), this was achieved using nltk.PorterStemmer. 


# Feature Vectorization
Once data preprocessing steps were completed, the cleaned text had to undergo vectorization, which is the process of converting text into an integer the model can recognize and analyze. “Bag of words” is the method used to convert the clean text into document-matrix that describes the frequency of each word in the text and CountVectorizer from sklearn.feature_extraction.text was utilized to vectorize and generate this document-matrix. In addition, CountVectorizer assisted in the transformation of the list of clean text into a string, which is needed for the split train and test set. 

# Build Machine Learning Model
At this stage of analysis, the clean text was ready to be deployed into the model. The predictor variable was clean text, and the target variable was the sentimental rating. The next step is cross-validation, designing split train (70%) and test (30%) set using train_test_split from sklearn. model_selection. In this analysis, the target variable was binary and logistic regression was a suitable model for sentimental analysis. The model was fitted using X_train and Y_train datasets, followed by predicting the output using the X_test dataset. 

# Model Evaluation 
F1 score was the selected metric to evaluate the model performance because it takes into account both precision and recall parameters to produce a score. F1 score was 0.88 for this model. This can be interpreted as we can be 88% confident the model identifies the right sentiment based on the clean review text. In addition, the model was able to show and identify the top 10 words associated with positive and negative sentiment.  Examples of the positive and negative sentiment words are shown below.
Positive sentiment words (great, amaz, love. delic, definit, friendly)
Negative sentiment words (worst, disappoint, ok, noth, rude)



Business value for yelp: Yelp can perform sentimental analysis for specific industries and businesses and charge service fee for the analytical solution. The analytical solution helps businesses understand their customer’s needs and how the customer feels about their product. It identifies keywords for advertising new products and provides clarity on the quality of the services and areas the business is underperforming. 



![image](https://user-images.githubusercontent.com/97130281/148303237-4795293f-b3e5-41ce-9c4d-e0efa71240e2.png)

