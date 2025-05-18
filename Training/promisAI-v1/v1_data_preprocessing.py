import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import seaborn as sns

data = [
    {
        "title": "a pre evaluation of the behavioural relationship of applied caoting systems under two expose condions phase ii natural weathering",
        "concept": [
            "weathering performance", "polyurethane lacquer", "latex enamel paint", 
            "natural weathering", "outdoor exposure", "tangible wood sample"
        ],
        "sdg": [
            "climate action", "responsible consumption production"
        ],
        "presidential_development_plan": [
            "infrastructure development", "energy security"
        ],
        "hnrda": [
            "industry energy emerge technology"
        ]
    },
]

df = pd.DataFrame(data)

def flatten_json(row):
    all_text = ' '.join(row['concept'] + row['sdg'] + row['presidential_development_plan'] + row['hnrda'])
    return all_text

df['all_text'] = df.apply(flatten_json, axis=1)

print(f"Total records: {len(df)}")
print(f"Info:\n{df.info()}")
print(f"Describe:\n{df.describe()}")
print(f"Head:\n{df.head()}")
print(f"Shape: {df.shape}")


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['all_text'])

words = vectorizer.get_feature_names_out()

word_freq = dict(zip(words, X.sum(axis=0).A1))

word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Frequency', y='Word', data=word_freq_df.head(20), palette="viridis")
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Frequent Words')
plt.show()

vectorizer_bigrams = CountVectorizer(ngram_range=(2, 2), stop_words='english')
X_bigrams = vectorizer_bigrams.fit_transform(df['all_text'])
bigram_freq = dict(zip(vectorizer_bigrams.get_feature_names_out(), X_bigrams.sum(axis=0).A1))

bigram_freq_df = pd.DataFrame(list(bigram_freq.items()), columns=['Bigram', 'Frequency'])
bigram_freq_df = bigram_freq_df.sort_values(by='Frequency', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Frequency', y='Bigram', data=bigram_freq_df.head(10), palette="Blues")
plt.title('Top 10 Bigrams (Two-Word Phrases)')
plt.xlabel('Frequency')
plt.ylabel('Bigrams')
plt.show()

