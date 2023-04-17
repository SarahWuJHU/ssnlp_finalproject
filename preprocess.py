import wikipedia
from gensim.models import Word2Vec
import nltk
import json
nltk.download('punkt')


def process_remove_metadata():
    # Set the language to English
    wikipedia.set_lang("en")
        
    # Set up a list of sections to exclude from each Wikipedia article
    excluded_sections = ['See also', 'References', 'Further reading', 'External links']

    # Set up a regular expression pattern to match markdown
    markdown_pattern = re.compile(r'(\'{2,5})(.*?)\1')

    # Set up a list to hold the cleaned text from each Wikipedia article
    cleaned_articles = []

    # Loop over a subset of Wikipedia articles
    article_titles = ['Python (programming language)', 'Machine learning', 'Data science']
    for article_title in article_titles:
        # Get the raw text of the Wikipedia article
        raw_article = wikipedia.page(article_title).content
        
        # Remove any markdown from the article text
        cleaned_article = re.sub(markdown_pattern, r'\2', raw_article)
        
        # Split the article text into sections
        sections = cleaned_article.split('\n==')

        # Loop over each section and exclude any sections in the excluded_sections list
        cleaned_sections = []
        for section in sections:
            if not any(section.startswith(f"\n{es}\n") for es in excluded_sections):
                # Tokenize the section into sentences using nltk
                sentences = nltk.sent_tokenize(section)
                # Append the cleaned sentences to the cleaned_sections list
                cleaned_sections.extend(sentences)

        # Join the cleaned sections back together into a single string
        cleaned_article = " ".join(cleaned_sections)
        # Append the cleaned article to the cleaned_articles list
        cleaned_articles.append(cleaned_article)
        
    
    # Tokenize the cleaned text into lists of words using nltk
    tokenized_articles = [nltk.word_tokenize(article) for article in cleaned_articles]

    # Train a Word2Vec model on the tokenized articles
    model = Word2Vec(tokenized_articles, size=100, window=5, min_count=5, workers=4)

    # Test the model by getting the most similar words to 'Johns Hopkins University'
    print(model.wv.most_similar('Johns Hopkins University'))

def process_neutral():
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

def process_emotional():
    
    # Load the EmpatheticDialogues dataset
    # wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
    
    # Extract the dataset
    # tar -xzf empatheticdialogues.tar.gz

    # Load the EmpatheticDialogues dataset
    with open('empatheticdialogues.json') as f:
        data = json.load(f)

    # Tokenize each utterance with an emotional label
    sentences = []
    for conversation in data:
        for utterance in conversation['utterances']:
            # Tokenize the utterance text into sentences using nltk
            sentence_list = nltk.sent_tokenize(utterance['text'])
            for sentence in sentence_list:
                # Append the tokenized sentence with its emotional label
                sentences.append((sentence, utterance['emotion']))

    # Print the tokenized sentences with their emotional labels
    for sentence, label in sentences:
        print(f"{label}: {sentence}")
