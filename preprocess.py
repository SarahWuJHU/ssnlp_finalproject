import wikipedia
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Set the language to English
wikipedia.set_lang("en")

# Get a list of English Wikipedia article titles
titles = wikipedia.page("List of articles every Wikipedia should have").links

# Get the content of each article and tokenize it into sentences
sentences = []
for title in titles:
    try:
        # Download the article content
        page = wikipedia.page(title)
        text = page.content

        # Tokenize the text into sentences using nltk
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            # Tokenize each sentence into words using nltk
            word_list = nltk.word_tokenize(sentence)
            sentences.append(word_list)

    except Exception as e:
        # If an error occurs, print the title of the article and the error message
        print(f"Error downloading '{title}': {e}")

# Train a Word2Vec model on the tokenized sentences
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

# Save the trained model to a file
model.save("wikipedia_word2vec.model")
