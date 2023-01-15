import math
import numpy as np
import pandas as pd


def get_top_n(sim_dict, N=20):       # for search_body
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
    return sorted([(doc_id,score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `document_len`.
class BM_25_from_index:
    """
        Best Match 25.
        ----------
        k1 : float, default 1.5

        b : float, default 0.75

        index: inverted index
        """

    def __init__(self, index, avg_document_len, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.document_len)
        self.idf = None
        self.average_DL = avg_document_len

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_documents(self, query_to_search, words, pls):
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
        candidates = []
        set_words = set(words)
        unique_q = np.unique(query_to_search)
        for term in unique_q:
            if term in set_words:
                current_list = (pls[words.index(term)])
                for item in current_list:
                    candidates.append(item[0])
        return np.unique(candidates)

    def search(self, query, N=20, query_words=None, query_pls=None):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        candidates = self.get_candidate_documents(query, query_words, query_pls)
        self.idf = self.calc_idf(query)
        document_score = self._score(query, candidates, query_words, query_pls)
        result = get_top_n(document_score, N)
        return result


    def helper_score(self, doc_id, all_tf, idf):
        document_len = self.index.document_len.get(doc_id, 0)
        frequency = all_tf.get(doc_id, 0)
        numerator = frequency * idf * (self.k1 + 1)
        denominator = frequency + self.k1 * (1 - self.b + self.b * document_len / self.average_DL)
        return numerator / denominator

    def _score(self, query, candidates, query_words=None, query_pls=None):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        docs_scores_df = pd.DataFrame({'docid': candidates})
        docs_scores_df["score"] = 0
        for term in query:
            try:
                diction_all_tf = dict(query_pls[query_words.index(term)])
                idf = self.idf.get(term, 0)
                docs_scores_df["score"] += docs_scores_df["docid"].apply(lambda docid: self.helper_score(docid, diction_all_tf, idf))
            except:
                pass
        return pd.Series(docs_scores_df["score"].values, index=docs_scores_df["docid"]).to_dict()
