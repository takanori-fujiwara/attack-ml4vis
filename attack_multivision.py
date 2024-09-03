import os
import pandas as pd
import numpy as np

import altair
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from multivision.utils.ChartRecommender import ChartRecommender
from multivision.utils.VegaLiteRender import VegaLiteRender

from utils.multivision_attack_utils import (
    load_pretrained_models, compute_columns_feature,
    gen_single_column_indices_to_chart_features,
    column_indices_to_chart_type_scores, get_chart_types,
    get_column_feature_names)


def recommend_charts(df,
                     recommender,
                     n_charts=3,
                     min_column_selection_score=0.05,
                     plot_charts=True):
    ## rank the results by the final_score
    recommended_charts = pd.DataFrame.from_records(recommender.charts)
    # TF: min_column_selection_score should be applied to single chart recommendation as well
    recommended_charts['over_min_column_selection_score'] = recommended_charts[
        'column_selection_score'] > min_column_selection_score

    recommended_charts = recommended_charts.sort_values(
        by=['over_min_column_selection_score', 'final_score'],
        ascending=[False, False])

    # using multiple chart recommendation
    # current_mv = []
    # recommended_charts = recommender.recommend_mv(
    #     mv_model,
    #     max_charts=n_charts,
    #     min_column_selection_score=min_column_selection_score)
    # recommended_charts = pd.DataFrame.from_records(
    #     recommended_charts).sort_values(by='final_score', ascending=False)

    print(recommended_charts)
    ## select the top charts and render it by VegaLiteRender
    if plot_charts:
        for i in range(n_charts):
            recommend_chart = recommended_charts.iloc[i]
            vr = VegaLiteRender(chart_type=recommend_chart['chart_type'],
                                columns=recommend_chart['fields'],
                                data=recommender.df.to_dict('records'))
            chart = altair.Chart.from_dict(vr.vSpec)
            chart.save('filename.html')
            os.system('open filename.html')

    return recommender, recommended_charts


(
    word_embedding_dict,
    column_score_model,
    chart_type_model,
    mv_model,
) = load_pretrained_models()

### Attack Example 1 (penguins)
# before attack
df = pd.read_csv('./data/penguins.csv')
recommender = ChartRecommender(df, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df, recommender, n_charts=5)

# after attack
df_adv = df.copy()
df_adv[''] = ''
df_adv[''][np.random.randint(df_adv.shape[0])] = ' '
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv, recommender)

### Attack Example 2 (gapminder)
# before attack
df = pd.read_csv('./data/gapminder.csv')
recommender = ChartRecommender(df, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df, recommender)

# after attack
df_adv = df.copy()
df_adv['life_expect'][np.random.randint(df_adv.shape[0])] = ' '
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv, recommender)


### Attack Example 3 (white-box attack)
def get_grad_chart_type(chart_type_model,
                        input_feats,
                        aiming_outputs,
                        loss_fn=F.cross_entropy,
                        device='cpu'):
    input_feats_ = input_feats.to(device)
    aiming_outputs = aiming_outputs.to(device)

    input_feats_.requires_grad = True
    outputs = chart_type_model(input_feats_)

    chart_type_model.zero_grad()
    cost = loss_fn(outputs, aiming_outputs).to(device)
    cost.backward()

    return input_feats_.grad


def get_grad_column_score(column_score_model,
                          input_feats,
                          aiming_outputs,
                          loss_fn=nn.MSELoss(),
                          device='cpu'):
    input_feats_ = Variable(torch.Tensor(input_feats)).to(device)
    aiming_outputs = aiming_outputs.to(device)

    input_feats_.requires_grad = True
    tmp_outputs, _ = column_score_model.lstm(input_feats_)
    outputs = column_score_model.linear(tmp_outputs.flatten())

    column_score_model.zero_grad()
    cost = loss_fn(outputs, aiming_outputs).to(device)
    cost.backward(retain_graph=True)

    return input_feats_.grad


def get_grad_table_chart_type(chart_type_model, input_feats, normalize=True):
    input_feats_ = Variable(torch.Tensor(input_feats[None, :, :])).float()
    aiming_chart_logits = chart_type_model(
        input_feats_).clone().detach().float()
    aiming_chart_logits[:, torch.argmax(aiming_chart_logits)] -= 50
    # aiming_outputs = torch.Tensor([[original_outputs.mean()] *
    #    original_outputs.shape[1]])
    grad_chart_type = get_grad_chart_type(chart_type_model, input_feats_,
                                          aiming_chart_logits)

    grad_table_chart_type = None
    for i in range(len(selected_columns)):
        col_feat_name_to_grad = {}
        col_feat_names = get_column_feature_names()
        for name, g in zip(col_feat_names, grad_chart_type[0, i, :]):
            col_feat_name_to_grad[name] = float(g)

        col_feat_name_to_grad = pd.DataFrame.from_dict(col_feat_name_to_grad,
                                                       orient='index')
        if grad_table_chart_type is None:
            grad_table_chart_type = col_feat_name_to_grad
        else:
            grad_table_chart_type = pd.concat(
                [grad_table_chart_type, col_feat_name_to_grad], axis=1)
    grad_table_chart_type.columns = df.columns[np.array(selected_columns)]

    if normalize:
        grad_table_chart_type /= np.abs(np.array(grad_table_chart_type)).max()

    return grad_table_chart_type


def get_grad_table_column_score(column_score_model,
                                input_feats,
                                normalize=True):
    input_feats_ = Variable(torch.Tensor(input_feats[None, :, :])).float()
    tmp, _ = column_score_model.lstm(input_feats_)
    aiming_score = column_score_model.linear(tmp.flatten()) - 0.2

    grad_column_score = get_grad_column_score(column_score_model, input_feats_,
                                              aiming_score)

    grad_table_column_score = None
    for i in range(len(selected_columns)):
        col_feat_name_to_grad = {}
        col_feat_names = get_column_feature_names()
        for name, g in zip(col_feat_names, grad_column_score[0, i, :]):
            col_feat_name_to_grad[name] = float(g)

        col_feat_name_to_grad = pd.DataFrame.from_dict(col_feat_name_to_grad,
                                                       orient='index')
        if grad_table_column_score is None:
            grad_table_column_score = col_feat_name_to_grad
        else:
            grad_table_column_score = pd.concat(
                [grad_table_column_score, col_feat_name_to_grad], axis=1)

    grad_table_column_score.columns = df.columns[np.array(selected_columns)]

    if normalize:
        grad_table_column_score /= np.abs(
            np.array(grad_table_column_score)).max()

    return grad_table_column_score


# penguins example
df = pd.read_csv('./data/penguins.csv')
recommender = ChartRecommender(df, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df, recommender)

topk = 3
set_of_selected_columns = recommended_charts['indices'].iloc[:topk]
single_column_indices_to_chart_features = gen_single_column_indices_to_chart_features(
    df, word_embedding_dict)
for selected_columns in set_of_selected_columns:
    input_feats = single_column_indices_to_chart_features(selected_columns)
    grad_table_chart_type = get_grad_table_chart_type(chart_type_model,
                                                      input_feats)
    grad_table_column_score = get_grad_table_column_score(
        column_score_model, input_feats)
    print('=======\nchart type\n=======\n', grad_table_chart_type)
    print('=======\ncol score\n=======\n', grad_table_column_score)

# from the above tables, the index order seems to contribute a lot to the prediction
df_adv = df.copy()
# columns = np.array(df_adv.columns.copy())
# np.random.shuffle(columns)
# df_adv = df_adv[columns]
df_adv = df_adv[[
    'Species', 'Sex', 'Island', 'Beak Length (mm)', 'Beak Depth (mm)',
    'Flipper Length (mm)', 'Body Mass (g)'
]]
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv, recommender)

# gapminder example
df = pd.read_csv('./data/gapminder.csv')
recommender = ChartRecommender(df, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df, recommender)

topk = 3
set_of_selected_columns = recommended_charts['indices'].iloc[:topk]
single_column_indices_to_chart_features = gen_single_column_indices_to_chart_features(
    df, word_embedding_dict)
for selected_columns in [(0, 4, 5)]:  # set_of_selected_columns:
    input_feats = single_column_indices_to_chart_features(selected_columns)
    grad_table_chart_type = get_grad_table_chart_type(chart_type_model,
                                                      input_feats)
    grad_table_column_score = get_grad_table_column_score(
        column_score_model, input_feats)
    print('=======\nchart type\n=======\n', grad_table_chart_type)
    print('=======\ncol score\n=======\n', grad_table_column_score)

# from the above tables, the index order seems to contribute a lot to the prediction
df_adv = df.copy()
# columns = np.array(df_adv.columns.copy())
# np.random.shuffle(columns)
# df_adv = df_adv[columns]
df_adv = df_adv[[
    'cluster', 'life_expect', 'fertility', 'year', 'pop', 'country'
]]
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv,
                                                   recommender,
                                                   n_charts=3)

# Just change column names. This one also influences the recommendation
df_adv = df.copy()
df_adv.columns = [
    'year', 'country', 'cluster', 'pop', 'life_expect', 'births_per_woman'
]
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv, recommender)

# this one draws no lines, etc. But I think this is Vega-Lite's bug when the field name has period.
df_adv.columns = [
    'year', 'country', 'cluster', 'pop', 'life expect.', 'fertility'
]
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv, recommender)
