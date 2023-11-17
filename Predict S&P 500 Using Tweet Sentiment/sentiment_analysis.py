from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def tweet_processing(tweet):
    """
    Replace any user mentions with "@user" and any URL with "http".

    Parameters:
    tweet (str): Tweet to be processed

    Returns:
    str: Tweet on which sentiment analysis is to be performed.
    """

    # Pre-processed tweet
    tweet_words = ["@user" if word.startswith("@") and len(
        word) > 1 else "http" if word.startswith("http") else word for word in tweet.split(" ")]
    
    return " ".join(tweet_words)



def classify(tweet):
    """
    Perform sentiment analysis on a tweet using the RoBERTa language model.

    Parameters:
    tweet (str): Tweet on which sentiment analysis is to be performed.

    Returns:
    dict: 
        Keys: "Negative", "Neutral", "Positive"
        Values: Corresponding values denoting proportionality (values add up to 1)
    """

    # Load model and tokeniser
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokeniser = AutoTokenizer.from_pretrained(roberta)

    # Sentiment analysis
    labels = ["Negative", "Neutral", "Positive"]
    encoded_tweet = tokeniser(tweet_processing(tweet), return_tensors="pt")
    #output = model(encoded_tweet["input_ids"], encoded_tweet["attention_mask"])
    output = model(**encoded_tweet)

    scores = softmax(output[0][0].detach().numpy())
    sentiment = {labels[i]:scores[i] for i in range(len(scores))}
  
    return sentiment



