import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import minmax_scale

import sys

sys.path.append("..")

from multivision.utils import featureExtractor
from multivision.model.encodingModel import ChartTypeLSTM, ScoreNetLSTM


def get_column_feature_names(indices=None):
    feature_names = np.array([
        'column_idx_normed',
        'dataType_normed',
        'aggrPercentFormatted',  # Proportion of cells having percent format
        'aggr01Ranged',  # Proportion of values ranged in 0-1
        'aggr0100Ranged',  # Proportion of values ranged in 0-100
        'aggrIntegers',  # Proportion of integer values
        'aggrNegative',  # Proportion of negative values
        'aggrBayesLikeSum',  # Aggregated Bayes feature
        'dmBayesLikeDimension',  # Bayes feature for dimension measure
        'commonPrefix',  # Proportion of most common prefix digit
        'commonSuffix',  # Proportion of most common suffix digit
        'keyEntropy',  # Entropy by values
        'charEntropy',  # Entropy by digits/chars
        'norm_range',  # data_features.get('range', 0),  # Values range
        'changeRate',  # Proportion of different adjacent values
        'partialOrdered',  # Maximum proportion of increasing or decreasing adjacent values
        'norm_var',  # data_features.get('variance', 0),  # Standard deviation
        'norm_cov',  # data_features.get('cov', 0),  # Coefficient of variation
        'cardinality',  # Proportion of distinct values
        'spread',  # Cardinality divided by range
        'major',  # Proportion of the most frequent value
        'benford',  # Distance of the first digit distribution to real-life average
        'orderedConfidence',  # Indicator of sequentiality
        'equalProgressionConfidence',  # confidence for a sequence to be equal progression
        'geometircProgressionConfidence',  # confidence for a sequence to be geometric progression
        'medianLength',  # median length of fields' records, 27.5 is 99% value
        'lengthStdDev',  # transformed length stdDev of a sequence
        'sumIn01',  # Sum the values when they are ranged 0-1
        'sumIn0100',  # Sum the values when they are ranged 0-100
        'absoluteCardinality',  # Absolute Cardinality, 344 is 99% value
        'skewness',
        'kurtosis',
        'gini',
        'nRows',  # Number of rows, 576 is 99% value
        'averageLogLength',
        *[f'dummy{i}' for i in range(11)],
        *[f'wordEmb{i}' for i in range(50)]
        # *[f'wordEmb{i}' for i in range(word_embedding_dict['apple'].shape[0])]
    ])

    if indices is None:
        return feature_names
    else:
        return feature_names[indices]


def get_chart_types(indices=None):
    chart_types = np.array([
        'area', 'bar', 'bubble', 'line', 'pie', 'radar', 'scatter', 'stock',
        'surface'
    ])
    if indices is None:
        return chart_types
    else:
        return chart_types[indices]


def load_pretrained_models(
        word_embedding_model_path='multivision/utils/en-50d-200000words.vec',
        column_score_model_path='multivision/trainedModel/singleChartModel.pt',
        chart_type_model_path='multivision/trainedModel/chartType.pt',
        mv_model_path='multivision/trainedModel/mvModel.pt'):
    ### Model preparation by following the demo in https://github.com/wowjyu/MultiVision/blob/master/Demo.ipynb
    device = torch.device('cpu')

    word_embedding_dict = {}
    with open(word_embedding_model_path) as file_in:
        lines = []
        for idx, line in enumerate(file_in):
            if idx == 0:  ## line 0 is invalid
                continue
            word, *features = line.split()
            word_embedding_dict[word] = np.array(features)

    column_score_model = ScoreNetLSTM(input_size=96,
                                      seq_length=4,
                                      batch_size=2,
                                      pack=True).to(device)
    column_score_model.load_state_dict(
        torch.load(column_score_model_path, map_location=device))
    column_score_model.eval()

    chart_type_model = ChartTypeLSTM(input_size=96,
                                     hidden_size=400,
                                     seq_length=4,
                                     num_class=9,
                                     bidirectional=True).to(device)
    chart_type_model.load_state_dict(
        torch.load(chart_type_model_path, map_location=device))
    chart_type_model.eval()

    mv_model = ScoreNetLSTM(input_size=9, seq_length=12).to(device)
    mv_model.load_state_dict(torch.load(mv_model_path, map_location=device))
    mv_model.eval()

    return word_embedding_dict, column_score_model, chart_type_model, mv_model


def compute_columns_feature(df, word_embedding_dict):

    def field_type(idx):
        if idx == 1:
            return "nominal"
        elif idx == 3 or idx == 7:
            return "temporal"
        elif idx == 5:
            return "quantitative"
        return ""

    feature_dict = []
    fields = []
    for cIdx, column_header in enumerate(df.columns):
        column_values = df[column_header].tolist()
        dataType, data_features = featureExtractor.get_data_features(
            column_values)
        embedding_features = featureExtractor.get_word_embeddings(
            column_header, word_embedding_dict)

        column_idx_normed = min(1.0, cIdx / 50)
        dataType_normed = dataType / 5

        feature = [
            column_idx_normed, dataType_normed, *data_features,
            *np.zeros(11).tolist(), *embedding_features
        ]

        feature = np.nan_to_num(np.array(feature), 0)
        feature_dict.append(feature.tolist())
        fields.append({
            "name": column_header,
            "index": cIdx,
            "type": field_type(dataType)
        })

    return np.array(feature_dict), fields


def gen_single_column_indices_to_chart_features(df,
                                                word_embedding_dict,
                                                max_seq_len=4):

    def single_column_indices_to_chart_features(column_indices):
        col_feats, fields = compute_columns_feature(df, word_embedding_dict)
        chart_feats = np.zeros((max_seq_len, col_feats.shape[1]))
        chart_feats[:len(column_indices), :] = col_feats[column_indices, :]

        return chart_feats

    return single_column_indices_to_chart_features


def generate_all_column_indices(n_columns, max_seq_len=4):
    result = []
    for i in range(min(n_columns, max_seq_len)):
        result += list(itertools.combinations(np.arange(n_columns), i + 1))
    return result


def df_to_column_scores(df, word_embedding_dict, column_score_model):
    single_column_indices_to_chart_features = gen_single_column_indices_to_chart_features(
        df, word_embedding_dict)

    all_column_indices = generate_all_column_indices(df.shape[1])
    all_chart_features = np.array([
        single_column_indices_to_chart_features(col_indices)
        for col_indices in all_column_indices
    ])

    column_score_model_input = nn.utils.rnn.pack_padded_sequence(
        Variable(torch.Tensor(all_chart_features)).float(),
        np.full(*all_chart_features.shape[:-1]),
        batch_first=True,
        enforce_sorted=False)
    column_scores = column_score_model(column_score_model_input)
    column_scores = minmax_scale(column_scores.detach().flatten().numpy())

    result = {}
    for col_indices, scores in zip(all_column_indices, column_scores):
        result[col_indices] = scores

    return result


def column_indices_to_chart_type_scores(column_indices,
                                        df,
                                        word_embedding_dict,
                                        chart_type_model,
                                        chart_types=[
                                            'area', 'bar', 'bubble', 'line',
                                            'pie', 'radar', 'scatter', 'stock',
                                            'surface'
                                        ]):
    single_column_indices_to_chart_features = gen_single_column_indices_to_chart_features(
        df, word_embedding_dict)
    chart_features = single_column_indices_to_chart_features(column_indices)
    chart_type_scores = chart_type_model(
        Variable(torch.Tensor([chart_features])).float()).detach().numpy()[0]

    result = {}
    for chart_type, scores in zip(chart_types, chart_type_scores):
        result[chart_type] = scores

    return result
