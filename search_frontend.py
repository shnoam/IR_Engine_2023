import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import BM_25_from_index as bm25
import nltk as nltk
from google.cloud import storage
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import inverted_index_gcp

nltk.download('stopwords')
from nltk.corpus import stopwords
import re

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))

# set up variables in order to run the on gcp
bucket_name = "316139070_206204588"
client = storage.Client('shamir-ilay-the-kings')  # add project
bucket = client.bucket(bucket_name)


def get_index_from_bucket(bucket, indexName):
    blob = storage.Blob(f'postings_gcp/{indexName}.pkl', bucket)
    with open(f'./{indexName}.pkl', "wb") as f:
        blob.download_to_file(f)
    return inverted_index_gcp.InvertedIndex.read_index('./', indexName)


def get_content_from_storage(bucket, file_name):
    blob = storage.Blob(f'{file_name}', bucket)
    with open(f'./{file_name}', "wb") as file_obj:
        blob.download_to_file(file_obj)
    with open(Path("./") / f'{file_name}', 'rb') as f:
        return pickle.load(f)

def find_average_DL(index):
    res = np.mean(list(index.document_len.values()))
    return res

# get each index from buckets
inverted_index_text = get_index_from_bucket(bucket, 'text_index')
inverted_index_title = get_index_from_bucket(bucket, 'title_index')
inverted_index_anchor = get_index_from_bucket(bucket, 'anchor_index')

# get pageview and pagerank from bucket
diction_docs_pagerank = get_content_from_storage(bucket, "norm_pagerank.pkl")
diction_docs_pageview = get_content_from_storage(bucket, "norm_pageviews.pkl")

average_DL_text = find_average_DL(inverted_index_text)
bm25_index = bm25.BM_25_from_index(inverted_index_text, average_DL_text)


######################### helper functions #######################

def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in english_stopwords]
    return list_of_tokens


def get_candidate_documents_tf(query, to_normalize, words, pls):
    """

    :param query: type  = list
    :param words: all terms in index
    :param pls:  list of pls (each psl = tups of doc_id and freq)
    :return: sorted list of tuples : [(doc_id,term_freq),....]
    """
    tf_docs = {}  # key : doc_id , value : term frequency (no matter what term is it )
    for term in np.unique(query):
        if term in words:
            posting_list_by_doc = pls[words.index(term)]
            for tup in posting_list_by_doc:
                doc_id = tup[0]
                if to_normalize:
                    tf = tup[1] / len(inverted_index_title.doc_id_title[doc_id])
                else:
                    tf = tup[1]
                tf_docs[doc_id] = tf_docs.get(doc_id, 0) + tf
    candidate_docs = []
    for doc_id, tf in tf_docs.items():
        candidate_docs.append((doc_id, tf))
    return sorted(candidate_docs, key=lambda tup: tup[1], reverse=True)[:50]

def generate_query_tfidf_vector(query_to_search, index):  # for search_body
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    unique = np.unique(query_to_search)
    Q = np.zeros(len(unique))
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log((len(index.document_len)) / (df + epsilon), 10)  # smoothing
            try:
                ind = np.where(unique == token)[0][0]
                Q[ind] = tf * idf
            except:
                pass
    return Q

def cosine_similarity(D, Q, index):  # for search_body
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    cosine_sim_dic = {}
    d_trans = np.transpose(D)
    numerator = np.dot(Q, d_trans)
    norma_Q = np.linalg.norm(Q)
    all_docs_cosine_sim = numerator / norma_Q
    for i, docid in enumerate(D.index):
        cosine_sim_dic[docid] = all_docs_cosine_sim[i] / (index.doc_id_to_norm[docid])  # numerator[i] /( index.doc_id_to_norm[doc_id] * norma_Q)
    return cosine_sim_dic

def get_top_n(sim_dict, N=50):  # for search_body
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    terms = np.unique(query_to_search)
    in_words = np.isin(terms, words)
    unique = terms[in_words]
    for term in unique:
        list_of_doc = pls[words.index(term)]
        normalized_tfidf = []
        for doc_id, freq in list_of_doc:
            if index.document_len[doc_id] == 0:
                normalized_tfidf.append((doc_id, 0))
            else:
                normalized_tfidf.append((doc_id, (freq / index.document_len[doc_id]) * math.log(
                    len(index.document_len) / index.df[term], 10)))
        for doc_id, tfidf in normalized_tfidf:
            candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the query.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """
    unique_query_to_search = np.unique(query_to_search)  # check unique and np
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), len(unique_query_to_search)))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = unique_query_to_search
    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf
    return D

def get_topN_for_query(query, index, words, pls, N=50):
    vector = generate_query_tfidf_vector(query, index)  # Q
    tfidf_matrix = generate_document_tfidf_matrix(query, index, words, pls)  # D
    query_top_n = get_top_n(cosine_similarity(tfidf_matrix, vector, index), N)  # calc this query top n using cosine sim
    return query_top_n

################ end helpers ##################
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)  # split query to tokens

    if len(tokenized_query) == 0:
        return jsonify(res)

    words_title, pls_title = zip(*inverted_index_title.posting_lists_iter('postings_gcp/title_index', tokenized_query))
    title_score_values = dict(get_candidate_documents_tf(tokenized_query, False, words_title, pls_title))
    title_weight = 4

    words_body, pls_body = zip(*inverted_index_text.posting_lists_iter('postings_gcp/text_index', tokenized_query))
    bm25_scores = bm25_index.search(tokenized_query, 50, words_body, pls_body)

    final_docs_scores = defaultdict(int)    # init dictionary to store scores

    for doc_id, bm25score in bm25_scores:       # update weight to bm25 and pageview
        pageview = diction_docs_pageview.get(doc_id, 1)

        final_docs_scores[doc_id] += (2 * bm25score * pageview) / (bm25score + pageview)

    for doc_id, score in title_score_values.items():        # update weight to title score
        final_docs_scores[doc_id] *= title_weight * score

    final_docs_scores = sorted([(doc_id, score) for doc_id, score in final_docs_scores.items()], key=lambda x: x[1],
                               reverse=True)[:20]  # 100
    for doc_id, score in final_docs_scores:
        res.append((int(doc_id), inverted_index_title.doc_id_title.get(doc_id, "")))

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    words_body, pls_body = zip(*inverted_index_text.posting_lists_iter('postings_gcp/text_index',tokenized_query))  # words = all words in the titles,pls = list of psl (each psl = tups of doc_id and freq)
    sorted_candidate_doc = get_topN_for_query(tokenized_query, inverted_index_text, words_body, pls_body)   # return top 20
    for doc, tf_term in sorted_candidate_doc:
        res.append((int(doc), inverted_index_text.doc_id_title[doc]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    words_title, pls_title = zip(*inverted_index_title.posting_lists_iter('postings_gcp/title_index',tokenized_query))  # words = all words in the titles,pls = list of psl (each psl = tups of doc_id and freq)
    sorted_candidate_doc = get_candidate_documents_tf(tokenized_query, False, words_title,pls_title)
    for doc, tf_term in sorted_candidate_doc:
        res.append((int(doc), inverted_index_title.doc_id_title[doc]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    words_anchor, pls_anchor = zip(*inverted_index_anchor.posting_lists_iter('postings_gcp/anchor_index',tokenized_query))  # words = all words in the titles,pls = list of psl (each psl = tups of doc_id and freq)
    sorted_candidate_doc = get_candidate_documents_tf(tokenized_query, False, words_anchor,pls_anchor)
    for doc, tf_term in sorted_candidate_doc:
        res.append((int(doc), inverted_index_anchor.doc_id_title.get(doc, "")))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for id in wiki_ids:
        res.append(float(diction_docs_pagerank.get(id, 0)))  # list of all scores
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for id in wiki_ids:
        res.append(diction_docs_pageview.get(id, 0))  # list of all scores
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
