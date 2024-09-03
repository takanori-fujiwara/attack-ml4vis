import os
import pandas as pd
import numpy as np

# before attack
from sklearn import datasets
from sklearn.preprocessing import scale

from umap.parametric_umap import ParametricUMAP

# dataset = datasets.load_iris()
dataset = datasets.load_wine()
# dataset = datasets.load_breast_cancer()
# dataset = datasets.load_digits()

X = dataset.data
y = dataset.target
feat_names = dataset.feature_names
label_names = dataset.target_names
X = scale(X)

embedder = ParametricUMAP(encoder=None).fit(X)
emb_X = embedder.transform(X)

df = pd.DataFrame()
df['class_name'] = label_names[y]
df['x'] = emb_X[:, 0]
df['y'] = emb_X[:, 1]

# torch related modules need to load after UMAP
import os
import altair
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from multivision.utils.ChartRecommender import ChartRecommender
from multivision.utils.VegaLiteRender import VegaLiteRender

from utils.multivision_attack_utils import (
    load_pretrained_models, gen_single_column_indices_to_chart_features,
    get_column_feature_names)
from utils.substitute_model import SubstituteDR


class SubstituteEncoder(nn.Module):

    def __init__(self, dims, n_components=2, hidden_size=50):
        super(SubstituteEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_components),
        )

    def forward(self, x):
        return self.encoder(x)


class AttrCoordNN(nn.Module):

    def __init__(self, n_attrs, encoder, mask=None, fixed_values=None):
        import copy
        super(AttrCoordNN, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.attr_value_learner = nn.Sequential(nn.Linear(1, n_attrs), )

        self.mask = torch.Tensor(np.ones((n_attrs))).int()
        if mask is not None:
            self.mask = torch.Tensor(mask).int()

        self.fixed_values = torch.Tensor(np.zeros((n_attrs))).float()
        if fixed_values is not None:
            torch.Tensor(fixed_values).float()

    def forward(self, x):
        return self.encoder(
            self.attr_value_learner(torch.eye(1)) * self.mask +
            self.fixed_values * (1 - self.mask))

    def generate_instance(self):
        return self.attr_value_learner(
            torch.eye(1)) * self.mask + self.fixed_values * (1 - self.mask)


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


def get_grad(model,
             input_features,
             aiming_outputs,
             loss_fn=nn.CrossEntropyLoss(),
             device='cpu'):
    input_features = input_features.to(device)
    aiming_outputs = aiming_outputs.to(device)

    input_features.requires_grad = True
    outputs = model(input_features)

    model.zero_grad()
    cost = loss_fn(outputs, aiming_outputs).to(device)
    cost.backward()

    return input_features.grad


(
    word_embedding_dict,
    column_score_model,
    chart_type_model,
    mv_model,
) = load_pretrained_models()

recommender = ChartRecommender(df, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df, recommender)

new_sample = pd.DataFrame(np.zeros((1, 2)), columns=['x', 'y'])
new_sample['class_name'] = 'class_0'

magnitudes = [1, 2, 4, 8, 16, 32, 64]
n_iter_per_mangitude = 10
found_successful_sample = False
for magnitude in magnitudes:
    print(magnitude)
    if found_successful_sample:
        break
    for i in range(n_iter_per_mangitude):
        new_sample['x'] = (np.random.rand(new_sample.shape[0]) -
                           0.5) * 2.0 * magnitude
        new_sample['y'] = (np.random.rand(new_sample.shape[0]) -
                           0.5) * 2.0 * magnitude

        df_new = pd.concat([df, new_sample])
        single_column_indices_to_chart_features = gen_single_column_indices_to_chart_features(
            df_new, word_embedding_dict)

        column_scores = []
        for selected_columns in [[0], [1], [2], [0, 1], [0, 2], [1, 2],
                                 [0, 1, 2]]:
            input_feats = single_column_indices_to_chart_features(
                selected_columns)
            input_feats = Variable(torch.Tensor(
                input_feats[None, :, :])).float().to('cpu')
            outputs, _ = column_score_model.lstm(input_feats)
            outputs = column_score_model.linear(
                outputs.flatten()).detach().numpy()
            column_scores.append(outputs[0])
        column_scores = np.array(column_scores)
        if np.argmax(column_scores) in [0, 1, 2, 3, 4]:
            found_successful_sample = True
            print('col score changed', new_sample)
            break

        input_feats = single_column_indices_to_chart_features((1, 2))
        input_feats = Variable(torch.Tensor(input_feats[None, :, :])).float()

        chart_logits = chart_type_model(input_feats).clone().detach().float()
        chart_logits[0][3] = 0.0  # line chart wil not be selected in any cases

        if chart_logits[0][6] != chart_logits.max():
            found_successful_sample = True
            print('chart type changed', new_sample)
            break

aiming_pos = np.array([new_sample['x'][0], new_sample['y'][0]])

# use substitute model to generate the above coords
subs_embedder = SubstituteDR(ref_emb=emb_X,
                             encoder=SubstituteEncoder(X.shape[1:]))
subs_embedder.fit(X, max_epochs=2000)

attr_coord_mapper = SubstituteDR(ref_emb=aiming_pos[None, :],
                                 encoder=AttrCoordNN(
                                     n_attrs=X.shape[1],
                                     encoder=subs_embedder.encoder),
                                 loss_fn=F.huber_loss,
                                 learning_rate=0.1)
attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=2000)
new_x = attr_coord_mapper.encoder.generate_instance().detach().cpu().numpy()
print(subs_embedder.transform(new_x))
print(embedder.transform(new_x))

emb_new_x = embedder.transform(new_x)
df_new_x = pd.DataFrame()
df_new_x['class_name'] = [-1] * emb_new_x.shape[0]
df_new_x['x'] = emb_new_x[:, 0]
df_new_x['y'] = emb_new_x[:, 1]

df_adv = pd.concat([df, df_new_x])
recommender = ChartRecommender(df_adv, word_embedding_dict, column_score_model,
                               chart_type_model)
recommender, recommended_charts = recommend_charts(df_adv,
                                                   recommender,
                                                   plot_charts=True)
