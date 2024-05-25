import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from flask import Flask, request, jsonify
from subprocess import check_output

diff=pd.read_csv("/health-recommendation-system\Datasets\Symptom checker\diffsydiw.csv")
sym=pd.read_csv("/health-recommendation-system\Datasets\Symptom checker\sym_t.csv")
dia=pd.read_csv("/health-recommendation-system\Datasets\Symptom checker\dia_t.csv")

dia['diagnose'] = dia['diagnose'].str.replace('\x0b', ': ')
#dia['diagnose'] = dia['diagnose'].str.replace('\x92', ': ')

sd_diff=diff.merge(sym, left_on='syd', right_on='syd')
sd_diff=sd_diff.merge(dia, left_on='did', right_on='did')
# print(sd_diff.head())
# dia.head(20)

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix

def read_data(filename):

    data = pd.read_csv(filename,
                             usecols=[0,1,2],       
                             names=['user', 'song','plays'],skiprows=1) 
    data=data.dropna(axis=0, how='any')  #drop nan
    data['plays']=data['plays']+1


    data['user'] = data['user'].astype("category")
    data['song'] = data['song'].astype("category")

    # create a sparse matrix of all the users/plays
    plays = coo_matrix((data['plays'].astype(float),
                       (data['song'].cat.codes.copy(),
                        data['user'].cat.codes.copy())))

    return data, plays,data.groupby(['song']).plays.sum(),data['user'].cat.codes.copy()

data,matrix,songsd,user=read_data('D:\MSc Data Analytics\Project\Individual Project\Datasets\Symptom checker\diffsydiw.csv')


#user=symptom
#sond=diagnose

from sklearn.preprocessing import normalize


def cosine(plays):
    normalized = normalize(plays)
    return normalized.dot(normalized.T)


def bhattacharya(plays):
    plays.data = np.sqrt(plays.data)
    return cosine(plays)


def ochiai(plays):
    plays = csr_matrix(plays)
    plays.data = np.ones(len(plays.data))
    return cosine(plays)


def bm25_weight(data, K1=1.2, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))

    # calculate length_norm per document
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret


def bm25(plays):
    plays = bm25_weight(plays)
    return plays.dot(plays.T)

def get_largest(row, N=10):
    if N >= row.nnz:
        best = zip(row.data, row.indices)
    else:
        ind = np.argpartition(row.data, -N)[-N:]
        best = zip(row.data[ind], row.indices[ind])
    return sorted(best, reverse=True)


def calculate_similar_artists(similarity, artists, artistid):
    neighbours = similarity[artistid]
    top = get_largest(neighbours)
    return [(artists[other], score, i) for i, (score, other) in enumerate(top)]


#songsd = dict(enumerate(data['song'].cat.categories))
user_count = data.groupby('user').size()
#to_generate = sorted(list(songsd), key=lambda x: -user_count[x])

similarity = bm25_weight(matrix)


sym[sym['syd'].isin(list(songsd.index))]

from scipy.sparse.linalg import svds

Ur, Si, VTr = svds(bm25_weight(coo_matrix(matrix)), k=100)

VTr=pd.DataFrame(VTr)

from sklearn.metrics.pairwise import cosine_similarity
Sddf=pd.DataFrame(cosine_similarity(Ur,VTr.T),columns=user_count.index,index=list(songsd.index))
#Sddf.to_csv("F:\Symptom checker")

Sydi=pd.DataFrame(cosine_similarity(Ur,VTr.T))

# Assuming you have loaded the 'prec_t.csv' dataset into a DataFrame named 'treatments'
# 'dia' DataFrame is assumed to contain 'did' and 'diagnose' columns as in the previous code
treatments= pd.read_csv('D:\MSc Data Analytics\Project\Individual Project\Datasets\Symptom checker\prec_t.csv')
# 1. Take symptoms as input from the user
# user_input = input("Enter symptoms separated by commas: ")
# user_symptoms = [symptom.strip() for symptom in user_input.split(',')]

# # 2. Recognize the indices of the input symptoms
# recognized_indices = []
# for user_symptom in user_symptoms:
#     # Assuming 'syd' is the column containing symptom indices in the 'sym' DataFrame
#     recognized_index = sym[sym['symptom'] == user_symptom]['syd'].values
#     if len(recognized_index) > 0:
#         recognized_indices.append(recognized_index[0])

def recommend_diseases(recognized_indices, Sddf, dia, treatments, sym):
    results = []

    if len(recognized_indices) > 0:
        # symptoms_info = []
        # for index in recognized_indices:
        #     symptom_name = sym[sym['syd'] == index]['symptom'].values[0]
        #     symptoms_info.append({'syd': index, 'symptom': symptom_name})

        # results.append({'recognized_symptoms': symptoms_info})

        combined_similarity = Sddf[recognized_indices].mean(axis=1)
        top_diseases = combined_similarity.sort_values(ascending=False)

        recommended_diseases = []
        for disease_id, similarity_score in top_diseases.head(5).items():
            diagnose = dia[dia['did'] == disease_id]['diagnose'].values[0]
            treatment = treatments[treatments['diagnose'] == diagnose]['pid'].values

            disease_info = {
                'diagnose': diagnose,
                'similarity_score': similarity_score,
                'recommended_treatment': treatment[0] if len(treatment) > 0 else None
            }
            recommended_diseases.append(disease_info)

        results.append({'recommended_diseases': recommended_diseases})

    return results



