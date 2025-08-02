# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-05)

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *
import libs.nnet.mmd as mmd
import torch.autograd as grad
from torch.nn.utils import clip_grad_norm_


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant=1.0):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


def gen_dist(sim_mat):
    mask = torch.eye(sim_mat.size(0), dtype=torch.bool).unsqueeze(1).repeat(1, sim_mat.shape[1], 1)
    # Use the mask to select the non-diagonal elements
    bc = sim_mat[~mask].view(sim_mat.shape[0], sim_mat.shape[1], sim_mat.shape[2] - 1)
    wc = sim_mat[mask].view(sim_mat.shape[0], sim_mat.shape[1], 1)
    num_samples = sim_mat.size(0) * sim_mat.size(1)
    return torch.cat((wc, bc), 2).view(num_samples, sim_mat.shape[-1]), wc.view(num_samples, 1), bc.view(num_samples,
                                                                                                 bc.shape[-1])

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utt_centroids(embeddings):
    sum_centroids = embeddings.sum(dim=1)
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    num_utterances = 2*embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utt_centroids(embeddings)
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1], -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances, -1
    )
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )

    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff, embeddings_flat, utterance_centroids_flat


def calc_loss(sim_mat):
    same_idx = list(range(sim_mat.size(0)))
    pos = sim_mat[same_idx, :, same_idx]
    neg = (torch.exp(sim_mat).sum(dim=2)+1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss


class MINE(torch.nn.Module):
    def __init__(self, emb_dims=512):
        super(MINE, self).__init__()
        self.l0 = nn.Linear(emb_dims, 512)
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(x)
        score = F.relu_(self.l0(x))
        score = F.relu_(self.l1(score))
        score = self.l2(score)
        score = torch.mean(score)
        return score


class TDNN(TopVirtualNnet):
    """ A standard x-vector framework """

    def init(self, inputs_dim, num_lang, nonlinearity="relu",
             margin_loss_params={}, pooling_params={},
             extracted_embedding="near"):

        # Var
        self.extracted_embedding = extracted_embedding

        # Nnet

        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim, 512, [-2, -1, 0, 1, 2], nonlinearity=nonlinearity)
        self.tdnn2 = ReluBatchNormTdnnLayer(512, 512, [-2, 0, 2], nonlinearity=nonlinearity)
        self.tdnn3 = ReluBatchNormTdnnLayer(512, 512, [-3, 0, 3], nonlinearity=nonlinearity)
        self.tdnn4 = ReluBatchNormTdnnLayer(512, 512, nonlinearity=nonlinearity)
        self.tdnn5 = ReluBatchNormTdnnLayer(512, 1500, nonlinearity=nonlinearity)


        self.lang_stats = MultiHeadAttentionPooling(1500, **pooling_params)
        self.spk_stats = MultiHeadAttentionPooling(1500, **pooling_params)

        self.lang_tdnn6 = ReluBatchNormTdnnLayer(self.spk_stats.get_output_dim(), 256, nonlinearity=nonlinearity)
        self.lang_tdnn7 = ReluBatchNormTdnnLayer(256, 256, nonlinearity=nonlinearity)
        self.lang_output = nn.Linear(256, num_lang)

        self.spk_tdnn6 = ReluBatchNormTdnnLayer(self.spk_stats.get_output_dim(), 256, nonlinearity=nonlinearity)
        self.spk_tdnn7 = ReluBatchNormTdnnLayer(256, 256, nonlinearity=nonlinearity)

        self.MINE = MINE(512)

    @utils.for_device_free
    def forward(self, inputs, alpha):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        outputs = []
        x = inputs
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        fspk = self.spk_stats(x)
        flang = self.lang_stats(x.detach())
        lang6 = self.lang_tdnn6(flang)
        lang = self.lang_tdnn7(lang6)
        lang_logit = self.lang_output(lang.view(-1, 256))
        outputs.append(lang_logit)

        spk6 = self.spk_tdnn6(fspk)
        spk = self.spk_tdnn7(spk6)
        outputs.append(spk)

        l_shuffle = torch.cat((lang[1:], lang[0].unsqueeze(0)), 0)
        ls = torch.cat((spk6.squeeze(), lang.squeeze()), 1)
        ls_shuffle = torch.cat((spk6.squeeze(), l_shuffle.squeeze().detach()), 1)
        reverse_ls = GradReverse.apply(ls, alpha)
        reverse_ls_shuffle = GradReverse.apply(ls_shuffle, alpha)

        pos = -F.softplus(-self.MINE(reverse_ls))
        neg = F.softplus(-self.MINE(reverse_ls_shuffle)) + self.MINE(reverse_ls_shuffle)
        mi = neg.mean() - pos.mean()

        outputs.append(mi)

        return outputs

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.spk_stats(x)

        if self.extracted_embedding == "far":
            xvector = self.spk_tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.spk_tdnn6(x)
            xvector = self.spk_tdnn7.affine(x)

        return xvector

#    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_lang_embs(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.lang_stats(x)

        x = self.lang_tdnn6(x)
        xvector = self.lang_tdnn7.affine(x)

        return xvector

class DSAN(TopVirtualNnet):
    def init(self, input_dims, num_targets, num_lang=106, nonlinearity="relu", num_per_class=4,
             emb_size=256, margin_loss=True, margin_loss_params={}, pooling_params={},
             tdnn_layer_params={}, extracted_embedding="near"):

        self.extracted_embedding = extracted_embedding
        self.num_lang = num_lang
        self.num_per_class = num_per_class
        self.emb_size = emb_size

        default_margin_loss_params = {
            "method": "am", "m": 0.2, "feature_normalize": True,
            "s": 30, "mhe_loss": False, "mhe_w": 0.01
        }

        default_pooling_params = {
            "num_head": 4,
            "hidden_size": 64,
            "share": True,
            "affine_layers": 1,
            "context": [0],
            "stddev": True,
            "temperature": False,
            "fixed": True
        }

        default_tdnn_layer_params = {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False, "bn": True, "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)


        self.feature_extractor = TDNN(input_dims, num_lang, nonlinearity=nonlinearity,
                                      margin_loss_params=margin_loss_params,
                                      pooling_params=pooling_params,
                                      extracted_embedding=self.extracted_embedding)

        self.lang_loss = nn.CrossEntropyLoss()
        self.loss_d = torch.nn.CrossEntropyLoss()
        self.loss = MarginSoftmaxLoss(256, num_targets, **margin_loss_params)

    @utils.for_device_free
    def do_gradient_ops(self):
        # Gradient scale
        self.w.grad *= 0.01
        self.b.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)


    @utils.for_device_free
    def forward(self, inputs, alpha):
        result = []
        lang_logit, spk_logit, mi = self.feature_extractor(inputs, alpha)
        result.append(lang_logit)
        result.append(spk_logit)
        result.append(mi)

        return result

    @utils.for_device_free
    def get_spk_loss(self, spk_mat):
        torch.clamp(self.w, 1e-6)
        return calc_loss(self.w * spk_mat + self.b)

    @utils.for_device_free
    def spk_mat(self, spk):
        num_spk = spk.shape[0] // self.num_per_class
        spk = spk / torch.norm(spk, dim=1).unsqueeze(1)
        spk_emb = spk.reshape(num_spk, self.num_per_class, self.emb_size)
        spk_centroids = get_centroids(spk_emb)
        spk_mat, _, _ = get_cossim(spk_emb, spk_centroids)
        return spk_mat

    @utils.for_device_free
    def get_loss(self, lang, spk, mi, targets, lang_labels):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        spk_loss = self.loss(spk, targets)
        lang_loss = self.lang_loss(lang, lang_labels)
        mi_loss = mi - torch.tensor(np.sqrt(2))

        return spk_loss, lang_loss, mi_loss

    @utils.for_device_free
    def get_closs(self, lang, spk, mi, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """

        spk_loss = self.loss(spk, targets)

        return spk_loss

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()


    @utils.for_device_free
    def extract_embedding(self, inputs):
        embedding = self.feature_extractor.extract_embedding(inputs)
        return embedding

    @utils.for_device_free
    def extract_lang_embs(self, inputs):
        embedding = self.feature_extractor.extract_lang_embs(inputs)
        return embedding

    def load_transform_state_dict(self, state_dict):
        assert isinstance(self.transform_keys, list)
        assert isinstance(self.rename_transform_keys, dict)

        remaining = {utils.key_to_value(self.rename_transform_keys, k, False): v for k, v in state_dict.items() if
                     k.split('.')[0] \
                     in self.transform_keys or k in self.transform_keys}
        self.load_state_dict(remaining, strict=False)

    def load_pretrained_module(self, initializer):
        pretrained_dict = torch.load(initializer)
#        loss_dict = self.state_dict()
#        train_loss_dict = {k: v for k, v in pretrained_dict.items() if k in loss_dict}
#        loss_dict.update(train_loss_dict)
#        self.load_state_dict(loss_dict, strict=False)

        model_dict = self.feature_extractor.state_dict()
        layer_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(layer_dict)
        self.feature_extractor.load_state_dict(model_dict, strict=False)
#        loss_dict.update()
        return self

