import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import numpy as np

def clean_sentences(sentences):
    sentences = [sentence for sentence in sentences if not pd.isna(sentence)]
    cleaned_sentences = []
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join(word for word in sentence.split() if not word.isdigit()) for sentence in sentences]
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_sentence = ' '.join(filtered_words)
        cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences

def remove_words(sentences, type): 
    del_words = ['months', 'years', 'days', 'since', 'left', 'right', 'breast', 'year', 'month']
    sentences = [' '.join([word for word in sentence.split() if word not in del_words]) for sentence in sentences]
    return sentences

def plot_most_frequent_words(sentences, type):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    words = vectorizer.get_feature_names_out()
    word_frequencies = X.sum(axis=0)
    word_freq_dict = dict(zip(words, word_frequencies.A1))
    sorted_word_freq = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    top_words = [word[0] for word in sorted_word_freq[:10]]
    top_word_freq = [word[1] for word in sorted_word_freq[:10]]

    sns.barplot(x=top_word_freq, y=top_words)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'{type}: Top 10 Most Frequently Occuring Bigrams')
    plt.tight_layout()
    # plt.savefig(f'{type}_basic_freq.png')
    plt.clf()

    return top_words, top_word_freq

def tfidf_words(sentences, type):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)
    top_words = tfidf_df.mean().sort_values(ascending=False).head(30)

    print(f'******** {type} ********')
    print("Top 10 Words and Their TF-IDF Scores:")
    words = []
    scores = []
    for word, score in top_words.items():
        # print(f"{word}: {score}")
        words.append(word)
        scores.append(score)

    return words[:30], scores[:30]

def normalize_freqs(array):
    return [(number/max(array) ) for number in array]

def create_excel(data, text, type):
    data_linear = []
    for el in data: 
        data_linear.append([el])

    text_linear = []
    for el in text: 
        text_linear.append([el])


    data_linear = np.array(data_linear)
    text_linear = np.array(text_linear)
    
    fig, ax = plt.subplots(figsize=(7, 7)) 

    start_color = "#808080"  # Light red 808080 ffb3b3
    end_color = "#333333"    # Slightly dark red 333333 b30000

    # start_color = "#009933"  # Light red
    # end_color = "#003311"    # Slightly dark red

    custom_palette = sns.blend_palette([start_color, end_color], as_cmap=True)
    ax = sns.heatmap(data_linear, 
                     cmap=custom_palette, 
                     annot=text_linear, fmt="", 
                     cbar=True, 
                     vmin=0, 
                     vmax=1, 
                    #  square=True,
                    #  annot_kws={'size': 24})
                     annot_kws={'size': 24, 'weight': 'bold'})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax.set_xticklabels([f'{type}'], rotation=0, weight='bold', fontsize=20)

    # ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    # plt.savefig(f"plot_bigram_{type}.png")
    plt.savefig(f"plot_unigram_{type}.png")
    plt.clf()
    # print(f'Word frequency data saved to plot_bigram_{type}.png.')
    print(f'Word frequency data saved to plot_unigram_{type}.png.')

def unique_10(top_words_mal, top_word_freq_mal, top_words_ben, top_word_freq_ben):
    return_words = []
    return_freqs = []

    for index, word in enumerate(top_words_mal):
        if word not in top_words_ben and len(return_words)<10:
            return_words.append(word)
            return_freqs.append(top_word_freq_mal[index])
    
    return return_words, return_freqs

def rep_words_table(top_words_mal, top_word_freq_mal, top_words_ben, top_word_freq_ben):
    top_words_mal, top_word_freq_mal = unique_10(top_words_mal, top_word_freq_mal, top_words_ben, top_word_freq_ben)
    top_words_ben, top_word_freq_ben = unique_10(top_words_ben, top_word_freq_ben, top_words_mal, top_word_freq_mal)
    # import pdb; pdb.set_trace()
    top_words_mal = top_words_mal[:10]
    top_word_freq_mal = top_word_freq_mal[:10]
    top_words_ben = top_words_ben[:10]
    top_word_freq_ben = top_word_freq_ben[:10]

    top_word_freq_mal = normalize_freqs(top_word_freq_mal)
    top_word_freq_ben = normalize_freqs(top_word_freq_ben)

    create_excel(top_word_freq_mal, top_words_mal, "MALIGNANT")
    create_excel(top_word_freq_ben, top_words_ben, "BENIGN")            

def make_bigram_freq(sentences, type):
    cv = CountVectorizer(ngram_range=(2,2))
    bigrams = cv.fit_transform(sentences)

    count_values = bigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["Frequency", "Bi-gram"]

    sns.barplot(x=ngram_freq['Frequency'][:20], y=ngram_freq['Bi-gram'][:20])
    plt.title(f'{type}: Top 10 Most Frequently Occuring Bigrams')
    plt.tight_layout()
    # plt.savefig(f'{type}_bigram.png')
    plt.clf()
    return ngram_freq['Bi-gram'][:10], ngram_freq['Frequency'][:10]

if __name__ == "__main__":
    test = "data_csv/test_correct.csv"
    train = "data_csv/train_correct.csv"

    # test = "data_csv/unprocessed_text/test_final.csv"
    # train = "data_csv/unprocessed_text/train_final.csv"

    df_test = pd.read_csv(test)
    df_train = pd.read_csv(train)

    labels_test = df_test['cancer'].to_list()
    labels_train = df_train['cancer'].to_list()
    text_mal = [text for index, text in enumerate(df_test['text'].to_list()) if labels_test[index]==1] + [text for index, text in enumerate(df_train['text'].to_list()) if labels_train[index]==1]
    text_ben = [text for index, text in enumerate(df_test['text'].to_list()) if labels_test[index]==0] + [text for index, text in enumerate(df_train['text'].to_list()) if labels_train[index]==0]

    text_mal = list(set(clean_sentences(text_mal)))
    text_ben = list(set(clean_sentences(text_ben)))

    type_mal = "MALIGNANT"
    type_ben = "BENIGN"

    text_mal = remove_words(text_mal, type_mal)
    text_ben = remove_words(text_ben, type_ben)

    # top_words_mal, top_word_freq_mal = plot_most_frequent_words(text_mal, type_mal)
    # top_words_ben, top_word_freq_ben = plot_most_frequent_words(text_ben, type_ben)

    top_words_mal, top_word_freq_mal = tfidf_words(text_mal, type_mal)
    top_words_ben, top_word_freq_ben = tfidf_words(text_ben, type_ben)

    # top_words_mal, top_word_freq_mal = make_bigram_freq(text_mal, type_mal)
    # top_words_ben, top_word_freq_ben = make_bigram_freq(text_ben, type_ben)

    rep_words_table(top_words_mal, top_word_freq_mal, top_words_ben, top_word_freq_ben)

    # print()
    # tfidf_words(text_mal, type_mal)
    # print()
    # tfidf_words(text_ben, type_ben)









