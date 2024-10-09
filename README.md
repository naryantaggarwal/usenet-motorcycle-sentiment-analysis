# Sentiment Analysis of Newsgroup Data

This project demonstrates a **sentiment analysis** of the **rec.motorcycles** category from the [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/). The program uses Natural Language Processing (NLP) techniques to preprocess the text and analyzes the sentiment of the documents in the selected category. The results are visualized using a histogram of sentiment scores.

### Workflow

1. **Data Loading**:
   - The **20 Newsgroups Dataset** is loaded using Scikit-learn's `fetch_20newsgroups` function. The category chosen for this analysis is `rec.motorcycles`.

2. **Text Preprocessing**:
   - **Tokenization**: The text is tokenized into words using NLTK's `word_tokenize`.
   - **Lowercasing**: The text is converted to lowercase to ensure consistency.
   - **Lemmatization**: Words are reduced to their base form (e.g., "running" becomes "run") using NLTK's `WordNetLemmatizer`.
   - **Stopword Removal**: Common English stopwords (e.g., "the", "is") are removed using NLTK's `stopwords`.

3. **Sentiment Analysis**:
   - The **VADER (Valence Aware Dictionary and sEntiment Reasoner)** sentiment analysis tool from NLTK is used to compute a **compound sentiment score** for each document. The compound score ranges from -1 (most negative) to +1 (most positive).

4. **Visualization**:
   - A **histogram** of sentiment scores is plotted using Matplotlib to show the distribution of positive and negative sentiments in the `rec.motorcycles` newsgroup documents.

### Installation

To run this program, you will need to install the following libraries:
```bash
pip install nltk scikit-learn matplotlib
```

Additionally, download the required NLTK datasets:
```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
nltk.download("punkt")
```

### How to Run

1. Load the script in a Python environment or Jupyter notebook.
2. The script will automatically fetch the **rec.motorcycles** data from the 20 Newsgroups Dataset.
3. It preprocesses the text, performs sentiment analysis, and displays a histogram of the sentiment scores.

### Example Output

The output will display a histogram showing the distribution of sentiment scores for the motorcycle-related newsgroup:

- **Sentiment Scores**:
  - Positive scores: indicate a positive sentiment.
  - Negative scores: indicate a negative sentiment.
  - Scores near 0: indicate neutral sentiment.

### Key Functions

- `preprocess_text(text)`: Preprocesses the input text by lowercasing, tokenizing, lemmatizing, and removing stopwords.
- `get_sentiment_scores(text)`: Returns the **compound sentiment score** of the input text using VADER.
  
### Visualization

The histogram visualizes the frequency of sentiment scores in the `rec.motorcycles` newsgroup, providing an overview of the overall sentiment distribution.

### Dependencies

- **NLTK**: For tokenization, lemmatization, stopword removal, and sentiment analysis.
- **Scikit-learn**: For loading the newsgroup dataset.
- **Matplotlib**: For visualizing the sentiment scores.

### License

This project is open-source and available under the MIT License.
