# here i have imported all necessary librearies 
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import re
import os
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

"""
THE STRUCTURE OF THS CODE IS AS FOLLOWS:

1. preprocessing ( i used nltk as it was serving better that spacy for this purpose because of its simplicity
2. semantic similarity (i used sentence-transformers as it is a state-of-the-art model for semantic similarity)
3. similarity calculation (i used fuzzywuzzy and difflib for calculating weighted  similarity scores)
4. input function (to take text from user and return the similarity score)


"""

"""data preprocessing code block here we tokenize, remove stopwords(shortcuts etc ) and other basic things ,"""
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)



"""to achieve SOTA accuracy i have used this model to grasp semantic meaning of text 
more details about this model can be found in this page https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"""


sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def preprocess_text(text):
    """Comprehensive text preprocessing"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # CONTRACTIONS TO EXPAND SHORTCUTS WE USE IN LANGUAGE EXAMPLE DON'T, CAN'T, I'M etc
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # TOKEINISNG
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) 
              for token in tokens 
              if token.lower() not in stop_words and len(token) > 1]
    
    return ' '.join(tokens)



"""after the cleaning now we can calculate similarity score using different methods  
PLEASE NOTE: i have used "weighted calculation" technique bcoz this is an unlabelled problem 
and while trying different combinations of weights i found the below combination to work best """




def calculate_similarity(text1, text2):
    """Calculate weighted similarity score (0-1)"""
    # Preprocess both texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    #here we calculate similarity score using the model we loaded above
    emb1 = sbert_model.encode(text1_clean, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2_clean, convert_to_tensor=True)
    sbert_sim = np.dot(emb1, emb2.T).item()
    
    """ I have used  fuzzuwuzzy library as from my past experience i know that while wokring on semantic data
    this library and its fucntions are very handy more details in this page 
            https://pypi.org/project/fuzzywuzzy/"""
            
            
    token_set_ratio = fuzz.token_set_ratio(text1_clean, text2_clean) / 100
    
    """this calculates the overlap of common words between the two texts"""
    tokens1 = text1_clean.split()
    tokens2 = text2_clean.split()
    common_words = set(tokens1) & set(tokens2)
    ctc_max = len(common_words) / (max(len(tokens1), len(tokens2)) + 1e-5)
    
    """this calculates the longest common subsequence ratio between the two texts"""
    lcs_ratio = SequenceMatcher(None, text1_clean, text2_clean).find_longest_match().size / (min(len(text1_clean), len(text2_clean)) + 1e-5)
    
    # after various hit and trials i found this to be working on various diverse types of inputs
    weighted_score = (0.50 * sbert_sim) + \
                     (0.30 * token_set_ratio) + \
                     (0.15 * ctc_max) + \
                     (0.05 * lcs_ratio)
    
    return max(0, min(1, weighted_score))


"""this will serve as input fucntion to ytake text from user and return the similarity scorEe"""
def get_similarity_score(text1: str, text2: str) -> float:
     
    return calculate_similarity(text1, text2)