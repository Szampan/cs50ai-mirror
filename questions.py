import nltk
import sys
import os
import string
from math import log

FILE_MATCHES = 2
SENTENCE_MATCHES = 2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = list(os.walk(directory))[0][2]
    files_content = {file: load_content(directory, file) for file in files}
    return files_content

def load_content(directory, file_name):
    file_dir = os.path.join(directory, file_name)
    with open(file_dir, encoding="utf8") as file:
        content = file.read()
        return content   # usunąć slice



def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation_and_stopwords = list(string.punctuation) + nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(document.lower())
    tokenized = [word for word in tokenized if word not in punctuation_and_stopwords]  
    tokenized = list(filter(lambda word: any(map(str.isalpha, word)), tokenized))

    return tokenized

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set(word for document in documents.values() for word in document)
    words_idfs = {word: log( len(documents) / count_docs_containing(word, documents) ) for word in words}
    return words_idfs

def count_docs_containing(word, documents):
    counter = 0
    for document in documents.values():
        if word in document:
            counter += 1
    return counter

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    print('TOP FILES')
    file_rank = {}
    for file in files:
        file_rank[file] = tf_idf_sum(query, files[file], idfs)
    top_files_list = sorted(file_rank, key=file_rank.get, reverse=True)[:n]
    return top_files_list

def tf_idf_sum(query, doc_words, idfs):
    sum = 0
    for query_word in query:
        sum += tf_idf(query_word, doc_words, idfs)
    return sum

def tf_idf(word, doc_words, idfs):
    occurances_num = doc_words.count(word)
    result = occurances_num * idfs[word]
    return result

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_rank = {}

    for sentence in sentences:
        sentence_rank[sentence] = (idf_sum(query, sentences[sentence], idfs), query_term_density(query, sentences[sentence]))

    top_sentences_list = sorted(sentence_rank, key=sentence_rank.get, reverse=True)[:n]       # dodać "query term" do key - może w sentence_rank wartością powinna być krotka(idf_sum, query_term)
    
    return top_sentences_list

def query_term_density(query, sentence):
    common_words = [word for word in query if word in sentence]
    density = len(common_words) / len(sentence)
    return density

def idf_sum(query, sentence, idfs):
    sum = 0
    
    for query_word in query:
        if query_word in sentence:
            sum += idfs[query_word]

    return sum

if __name__ == "__main__":
    main()
 