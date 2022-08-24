import pandas as pd
from tqdm import tqdm
from utils.clean import strip_emoji,decontract,strip_all_entities, clean_hashtags,filter_chars,remove_mult_spaces,stemmer

csv_path = "F:\cyberbullying\cyberbullying_tweets.csv"

def load_csv():
    df = pd.read_csv(csv_path)
    df = df[~df.duplicated()]
    df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'type'})
    return df 

#Then we apply all the defined functions in the following order
def deep_clean(text):
    # remove emoji
    text = strip_emoji(text)
    #remove contractions
    text = decontract(text)
    #Remove punctuations, links, stopwords, mentions and \r\n new line characters
    # Stopwords, 由于一些常用字或者词使用的频率相当的高，英语中比如a，the, he等，中文中比如：我、它、个等，每个页面几乎都包含了这些词汇，如果搜索引擎它们当关键字进行索引，那么所有的网站都会被索引，而且没有区分度，所以一般把这些词直接去掉，不可当做关键词。
    text = strip_all_entities(text)
    #clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the "#" symbol
    text = clean_hashtags(text)
    #Filter special characters such as "&" and "$" '#' present in some words
    text = filter_chars(text)
    #Remove multiple sequential spaces
    text = remove_mult_spaces(text)
    #Stemming
    output = stemmer(text)
    
    # if output!=text:
    #     print(f"output: {output}\ntext: {text}")

    return output


def main():
    data = load_csv()
    # data.info()
    # print(data.type.value_counts())

    texts_new = []

    with tqdm(total=len(data.text)) as pbar:
        for t in data.text:
            texts_new.append(deep_clean(t))
            pbar.update(1)
    
    data["text_clean"] = texts_new
    data.head()

    print(data["text_clean"].duplicated().sum())

    data.drop_duplicates("text_clean", inplace=True)

    print(data.type.value_counts())

    text_len = []
    for text in data.text_clean:
        if type(text) == str: 
            tweet_len = len(text.split())
            text_len.append(tweet_len)
        else:
            text_len.append(0)

    data['text_len'] = text_len
    data = data[data['text_len'] > 2]
    data = data[data['text_len'] < 100]

    # print(data.sort_values(by=["text_len"], ascending=False))
    data['type'] = data['type'].replace({"not_cyberbullying":0,"religion":1,"age":2,"ethnicity":3,"gender":4,"other_cyberbullying":5})


    data.to_csv("F:\cyberbullying\cyberbullying_tweets_clean_all.csv", encoding="utf-8", index=False)

    training_val_data = data.sample(frac=0.9)
    training_data = training_val_data.sample(frac=0.9)
    val_data = training_val_data[~training_val_data.index.isin(training_data.index)]
    test_data = data[~data.index.isin(training_val_data.index)]
    
    training_val_data.to_csv("F:\cyberbullying\cyberbullying_tweets_clean_training_val_data.csv", encoding="utf-8", index=False)
    training_data.to_csv("F:\cyberbullying\cyberbullying_tweets_clean_training_data.csv", encoding="utf-8", index=False)
    test_data.to_csv("F:\cyberbullying\cyberbullying_tweets_clean_test_data.csv", encoding="utf-8", index=False)
    val_data.to_csv("F:\cyberbullying\cyberbullying_tweets_clean_val_data.csv", encoding="utf-8", index=False)

if __name__ == "__main__":
    main()