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
import libs.nnet.mmd as mmd



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
    num_spk= wc.shape[0]
    num_per_class = wc.shape[1]
    for i in range(wc.shape[0]):
        if i == 0:
            bc_new = bc[0, :num_per_class].view(1, (num_spk-1)*num_per_class)
            wc_new = wc[0, :num_per_class].view(1, num_per_class)
        else:
            bc_new = torch.cat((bc_new, bc[i, :num_per_class].view(1, (num_spk-1)*num_per_class)), 0)
            wc_new = torch.cat((wc_new, wc[i, :num_per_class].view(1, num_per_class)), 0)

    return torch.cat((wc_new, bc_new), 1), wc_new.view(num_spk, wc_new.shape[-1]), bc_new.view(num_spk, bc_new.shape[-1])


def get_centroids(embeddings):
    # Calculating centroids for each speaker in the batch
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


def get_covmat(embeddings):
    utterance_centroids = get_utt_centroids(embeddings)
    centered_embs = embeddings - utterance_centroids
    embs = centered_embs.view(utterance_centroids.shape[0] * utterance_centroids.shape[1], -1)
    covmat = torch.cov(embs)
    return covmat


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

    same_speaker = torch.eye(embeddings.shape[0]).to(embeddings.device)
    same_speaker = same_speaker.repeat_interleave(num_utterances, dim=0)
    # Adjust the diagonal (i-th utterance compared to i-th centroid should be skipped)
#    same_speaker[torch.eye(num_speakers * num_utterances).bool().to(embeddings.device)] = 0
#    cos_sim_matrix[torch.eye(num_speakers * num_utterances).bool().to(embeddings.device)] = 0

#    return cos_sim_matrix.view(-1), same_speaker.view(-1)
    return cos_diff, same_speaker.view(embeddings.shape[0]*num_utterances, -1)



class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, cos_sim_matrix, labels):
        # embeddings shape: (speakers_per_batch, utterances_per_speaker, embedding_size)

        torch.clamp(self.w, 1e-6)

        # Scaling cosine similarities
        cos_sim_matrix = self.w * cos_sim_matrix + self.b

        # Applying softmax and calculating the loss
        loss = F.binary_cross_entropy_with_logits(cos_sim_matrix, labels)

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

    def init(self, inputs_dim, num_lang, num_dom, num_spk=32, num_utt=4, nonlinearity="relu",
             margin_loss_params={}, pooling_params={}, tdnn_layer_params={},
             extracted_embedding="near"):

        # Var
        self.extracted_embedding = extracted_embedding
        self.num_spk = num_spk
        self.num_utt = num_utt

        # Nnet

        self.tdnn1 = ReluBatchNormTdnnLayer(inputs_dim, 512, [-2, -1, 0, 1, 2], **tdnn_layer_params)
        self.tdnn2 = ReluBatchNormTdnnLayer(512, 512, [-2, 0, 2], **tdnn_layer_params)
        self.tdnn3 = ReluBatchNormTdnnLayer(512, 512, [-3, 0, 3], **tdnn_layer_params)
        self.tdnn4 = ReluBatchNormTdnnLayer(512, 512, **tdnn_layer_params)
        self.tdnn5 = ReluBatchNormTdnnLayer(512, 1500, **tdnn_layer_params)


        self.lang_stats = MultiHeadAttentionPooling(1500, **pooling_params)
        self.spk_stats = MultiHeadAttentionPooling(1500, **pooling_params)
 #       self.dom_stats = MultiHeadAttentionPooling(1500, **pooling_params)

        self.lang_tdnn6 = ReluBatchNormTdnnLayer(self.spk_stats.get_output_dim(), 256, **tdnn_layer_params)
        self.lang_tdnn7 = ReluBatchNormTdnnLayer(256, 256, **tdnn_layer_params)
        self.lang_output = nn.Linear(256, num_lang)

        self.spk_tdnn6 = ReluBatchNormTdnnLayer(self.spk_stats.get_output_dim(), 256, **tdnn_layer_params)
        self.spk_tdnn7 = ReluBatchNormTdnnLayer(256, 256, **tdnn_layer_params)

#        self.dom_tdnn6 = ReluBatchNormTdnnLayer(self.spk_stats.get_output_dim(), 256, **tdnn_layer_params)
#        self.dom_tdnn7 = ReluBatchNormTdnnLayer(256, 256, **tdnn_layer_params)
#        self.dom_output = nn.Linear(256, num_dom)

        self.MINE = MINE(512)
#        self.MINE_dom = MINE(512)

        self.discriminator1 = nn.Linear(256, 128)
        self.discriminator2 = nn.Linear(128, 2)

    @utils.for_device_free
    def forward(self, inputs, lang_alpha, dom_alpha):
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
        branch = x.detach()
        fspk = self.spk_stats(x)
        flang = self.lang_stats(branch)
  #      fdom = self.dom_stats(x.detach())
        lang6 = self.lang_tdnn6(flang)
        lang = self.lang_tdnn7(lang6)
        lang_logit = self.lang_output(lang.view(-1, 256))
        outputs.append(lang_logit)

        spk6 = self.spk_tdnn6(fspk)
        spk = self.spk_tdnn7(spk6)
        outputs.append(spk6)
        outputs.append(spk)

#        dom = self.dom_tdnn6(fspk.detach())
#        dom = self.dom_tdnn7(dom)
#        dom_logit = self.dom_output(dom.view(-1, 256))
#        outputs.append(dom_logit)

#        l_shuffle = torch.cat((lang[1:], lang[0].unsqueeze(0)), 0)
#        ls = torch.cat((spk6.squeeze(), lang.squeeze().detach()), 1)
#        ls_shuffle = torch.cat((spk6.squeeze(), l_shuffle.squeeze().detach()), 1)
#        reverse_ls = GradReverse.apply(ls, alpha)
#        reverse_ls_shuffle = GradReverse.apply(ls_shuffle, alpha)

#        spk_s = spk6.squeeze()
        lang_detach = lang.squeeze().detach()
        s_spk_s = spk6.view(self.num_spk, self.num_utt, -1)
        s_shuffle = torch.cat((s_spk_s[1:], s_spk_s[0].unsqueeze(0)),0)
        s_shuffle = s_shuffle.view(self.num_spk*self.num_utt, -1)
#        s_shuffle = torch.cat((spk6[1:], spk6[0].unsqueeze(0)), 0)
        sl = torch.cat((spk6.squeeze(), lang_detach), 1)
        sl_shuffle = torch.cat((s_shuffle.squeeze(), lang_detach), 1)
        reverse_sl = GradReverse.apply(sl, dom_alpha)
        reverse_sl_shuffle = GradReverse.apply(sl_shuffle, dom_alpha)

        pos_lang = -F.softplus(-self.MINE(reverse_sl))
        neg_lang = F.softplus(-self.MINE(reverse_sl_shuffle)) + self.MINE(reverse_sl_shuffle)
        mi_lang = neg_lang.mean() - pos_lang.mean()

 #       sd = torch.cat((spk6.squeeze(), dom.squeeze().detach()), 1)
 #       sd_shuffle = torch.cat((s_shuffle.squeeze(), dom.squeeze().detach()), 1)
 #       reverse_sd = GradReverse.apply(sd, lang_alpha)
 #       reverse_sd_shuffle = GradReverse.apply(sd_shuffle, lang_alpha)

 #       pos_dom = -F.softplus(-self.MINE(reverse_sd))
 #       neg_dom = F.softplus(-self.MINE(reverse_sd_shuffle)) + self.MINE(reverse_sd_shuffle)
 #      mi_dom = neg_dom.mean() - pos_dom.mean()

        outputs.append(mi_lang)
 #       outputs.append(mi_dom)

        reverse_emb = GradReverse.apply(spk.squeeze(), dom_alpha)
        flogits = F.relu_(self.discriminator1(reverse_emb))
        flogits = self.discriminator2(flogits)
        outputs.append(flogits)

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
        x = self.lang_stats(x)

        if self.extracted_embedding == "far":
            xvector = self.lang_tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.lang_tdnn6(x)
            xvector = self.lang_tdnn7.affine(x)

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
    def init(self, input_dims, num_targets, num_lang=106, num_dom=11, num_spk=64, num_utt=4,
             nonlinearity="relu", num_per_class=4, emb_size=256, margin_loss=True,
             margin_loss_params={}, pooling_params={}, tdnn_layer_params={},
             extracted_embedding="far"):

        self.extracted_embedding = extracted_embedding
        self.num_lang = num_lang
        self.num_spk = num_spk
        self.num_utt = num_utt
        self.num_per_class = num_per_class
        self.emb_size = emb_size
        self.num_dom = num_dom

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
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)

        self.feature_extractor = TDNN(input_dims, num_lang, num_dom, num_spk=self.num_spk, num_utt=self.num_utt, nonlinearity=nonlinearity,
                                      margin_loss_params=margin_loss_params,
                                      pooling_params=pooling_params, tdnn_layer_params=tdnn_layer_params,
                                      extracted_embedding=self.extracted_embedding)

#        self.discriminator1 = nn.Linear(256, 256)
#        self.discriminator2 = nn.Linear(256, 2)

        self.w = grad.Variable(torch.tensor(10.0))
        self.b = grad.Variable(torch.tensor(-5.0))

        self.lang_loss = nn.CrossEntropyLoss()
        self.mmd_loss = mmd.MaximumMeanDiscrepancy()
#        self.d_loss = MMD_loss()
        self.loss_d = torch.nn.CrossEntropyLoss()
#        self.loss_ds = torch.nn.MSELoss()
#        self.loss_dt = torch.nn.MSELoss()
 #       self.dom_loss = torch.nn.CrossEntropyLoss()
        self.loss = MarginSoftmaxLoss(256, num_targets, **margin_loss_params)
        self.ge2e_loss = GE2ELoss(init_w=10.0, init_b=-5.0)

    @utils.for_device_free
    def forward(self, source_inputs, target_inputs, lang_alpha, dom_alpha):
        result = []
#        inputs = torch.cat((source_inputs, target_inputs), 0)
#        lang_logit, spk_logits, dom_logit, mi_lang, mi_dom = self.feature_extractor(inputs, lang_alpha, dom_alpha)
#        spk_logit = spk_logits[:source_inputs.shape[0]]

        slang_logit, sspk, sspk_logit, smi_lang, sflogits = self.feature_extractor(source_inputs, lang_alpha, dom_alpha)
        tlang_logit, tspk, tspk_logit, tmi_lang, tflogits = self.feature_extractor(target_inputs, lang_alpha, dom_alpha)
 #       slang_logit, sspk, spk_logit, smi_lang = self.feature_extractor(source_inputs, lang_alpha, dom_alpha)
 #       tlang_logit, tspk, _, tmi_lang = self.feature_extractor(target_inputs, lang_alpha, dom_alpha)
        lang_logit = torch.cat((slang_logit, tlang_logit), 0)
  #      dom_logit = torch.cat((sdom_logit, tdom_logit), 0)
        mi_lang = 0.5*smi_lang + 0.5*tmi_lang
  #      mi_dom = 0.5*smi_dom + 0.5*tmi_dom
        flogits = torch.cat((sflogits, tflogits), 0)
#        spk = torch.cat((sspk, tspk), 0)
#        reverse_semb = GradReverse.apply(sspk.view(-1, self.emb_size), dom_alpha)
#        s_flogits = F.relu_(self.discriminator1(reverse_semb))
#        flogits = F.relu_(self.discriminator1(spk.view(-1, self.emb_size).detach()))
#        s_flogits = self.discriminator2(s_flogits)
#        print(flogits)
#        mmd_loss = self.d_loss(sspk.squeeze(), tspk.squeeze())

#        spk = torch.cat((sspk, tspk), 0)


        simmat_source, ssim_lbl = self.spk_mat(sspk_logit.squeeze().detach())
        simmat_target, tsim_lbl = self.spk_mat(tspk_logit.squeeze())


        sdist, swc, sbc = gen_dist(simmat_source)
        tdist, twc, tbc = gen_dist(simmat_target)

        loss_mmd_wc, loss_mmd_bc = self.mmd_loss(swc, twc, sbc, tbc)
        loss_mmd = loss_mmd_wc + 2 * loss_mmd_bc

        result.append(lang_logit)
        result.append(sspk_logit)
#        result.append(dom_logit)
        result.append(mi_lang)
#        result.append(mi_dom)
#        result.append(mmd_loss)
        result.append(flogits)
#        result.append(self.ge2e_loss(simmat_source.view(simmat_source.shape[0]*simmat_source.shape[1], -1), ssim_lbl))
        result.append(self.ge2e_loss(simmat_target.view(simmat_target.shape[0]*simmat_target.shape[1], -1), tsim_lbl))
        result.append(loss_mmd)

        return result

    @utils.for_device_free
    def spk_mat(self, spk):
        num_spk = spk.shape[0] // self.num_per_class
        spk = spk / torch.norm(spk, dim=1).unsqueeze(1)
        spk_emb = spk.reshape(num_spk, self.num_per_class, self.emb_size)
        spk_centroids = get_centroids(spk_emb)
        spk_mat, spk_lbl = get_cossim(spk_emb, spk_centroids)
        return spk_mat, spk_lbl

    @utils.for_device_free
    def get_loss(self, lang, spk, mi_lang, flogits, tspk_loss, loss_mmd, targets, lang_labels,
                 domains):
 #   def get_loss(self, lang, spk, mi_lang, flogits, sspk_loss, tspk_loss, loss_mmd, targets, lang_labels, dom_labels, domains):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        spk_loss = self.loss(spk, targets)
        lang_loss = self.lang_loss(lang, lang_labels)
 #       dom_loss = self.dom_loss(dom, dom_labels)
        lang_mi = mi_lang - torch.log(torch.tensor(4.0))
 #       dom_mi = mi_dom - torch.tensor(np.sqrt(2))
        loss_d = self.loss_d(flogits, domains) - torch.log(torch.tensor(2.0))
 #       loss_ge2e = sspk_loss + tspk_loss
 #       loss_ds = 0.5*self.loss_ds(flogits[:spk.shape[0]].squeeze(), domains[:spk.shape[0]])
 #       loss_dt = 0.5*self.loss_dt(flogits[spk.shape[0]:].squeeze(), domains[spk.shape[0]:]) #- torch.tensor(np.sqrt(2))/2.0
 #       loss_d = loss_ds + loss_dt - 0.25
 #       return spk_loss, lang_loss, lang_mi, loss_d, loss_ge2e, loss_mmd
        return spk_loss, lang_loss, lang_mi, loss_d, tspk_loss, loss_mmd

    @utils.for_device_free
    def get_closs(self, lang, spk, mi_lang, flogits, tspk_loss, loss_mmd, targets):
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
        model_dict = self.state_dict()
        layer_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(layer_dict)

#        dict_list = []

#        for key in pretrained_dict:
#            if 'batchnorm' in key:
                # Extract the corresponding weights and bias
#                bn_value = pretrained_dict[key]
#                new_key0 = key.replace("batchnorm", "batchnorm.bn_layers.domain_0")
#                new_key1 = key.replace("batchnorm", "bathcnorm.bn_layers.domain_1")
#                dict_list.append(new_key0)
#                dict_list.append(bn_value)
#                dict_list.append(new_key1)
#                dict_list.append(bn_value)

                # Assign the BN parameters to each domain-specific BN layer
#                for domain_key in model_dict:
#                    if 'batchnorm.bn_layers.domain_' in domain_key: #and key.replace('batchnorm', '') in domain_key:
#                        # Make sure to match the weights and biases to the correct domain-specific BN layer
#                        model_dict[domain_key] = bn_value.clone()
#        bn_dict = self.convert(dict_list)
#        model_dict.update(bn_dict)
        self.load_state_dict(model_dict, strict=False)
        return self

#    def load_pretrained_module(self, initializer):
#        pretrained_dict = torch.load(initializer)
#        loss_dict = self.state_dict()
#        train_loss_dict = {k: v for k, v in pretrained_dict.items() if k in loss_dict}
#        loss_dict.update(train_loss_dict)
#        self.load_state_dict(loss_dict, strict=False)

#        model_dict = self.feature_extractor.state_dict()
#        layer_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#        model_dict.update(layer_dict)
#        self.feature_extractor.load_state_dict(model_dict, strict=False)
#        loss_dict.update()
#        return self

