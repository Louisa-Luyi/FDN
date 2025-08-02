# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-28)

import sys
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *
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


class ResNetXvector(TopVirtualNnet):
    """ A resnet x-vector framework """

    def init(self, inputs_dim, num_targets, num_lang=106, num_dom=11, aug_dropout=0., tail_dropout=0., training=True, extracted_embedding="near",
             cmvn=False, cmvn_params={},
             resnet_params={}, pooling="statistics", pooling_params={}, fc1=True, fc1_params={}, fc2_params={},
             margin_loss=False, margin_loss_params={},
             use_step=False, step_params={}, transfer_from="softmax_loss", jit_compile=False):

        ## Params.
        default_cmvn_params = {
            "mean_norm": True,
            "std_norm": False,
        }

        default_resnet_params = {
            "head_conv": True, "head_conv_params": {"kernel_size": 3, "stride": 1, "padding": 1},
            "head_maxpool": False, "head_maxpool_params": {"kernel_size": 3, "stride": 1, "padding": 1},
            "block": "BasicBlock",
            "layers": [3, 4, 6, 3],
            "planes": [32, 64, 128, 256],  # a.k.a channels.
            "use_se": False,
            "se_ratio": 4,
            "convXd": 2,
            "norm_layer_params": {"momentum": 0.5, "affine": True},
            "full_pre_activation": True,
            "zero_init_residual": False
        }

        default_pooling_params = {
            "num_head": 1,
            "hidden_size": 64,
            "share": True,
            "affine_layers": 1,
            "context": [0],
            "stddev": True,
            "temperature": False,
            "fixed": True
        }

        default_fc_params = {
            "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}
        }

        default_margin_loss_params = {
            "method": "am", "m": 0.2, "feature_normalize": True,
            "s": 30, "mhe_loss": False, "mhe_w": 0.01
        }

        default_step_params = {
            "margin_warm": False,
            "margin_warm_conf": {"start_epoch": 5., "end_epoch": 10., "offset_margin": -0.2, "init_lambda": 0.0},
            "T": None,
            "m": False, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
            "s": False, "s_tuple": (30, 12), "s_list": None,
            "t": False, "t_tuple": (0.5, 1.2),
            "p": False, "p_tuple": (0.5, 0.1)
        }
        cmvn_params = utils.assign_params_dict(default_cmvn_params, cmvn_params)
        resnet_params = utils.assign_params_dict(default_resnet_params, resnet_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        ## Var.
        self.extracted_embedding = extracted_embedding  # only near here.
        self.use_step = use_step
        self.step_params = step_params
        self.convXd = resnet_params["convXd"]

        ## Nnet.
        self.aug_dropout = torch.nn.Dropout2d(p=aug_dropout) if aug_dropout > 0 else None
        self.cmvn_ = InputSequenceNormalization(**cmvn_params) if cmvn else torch.nn.Identity()

        # [batch, 1, feats-dim, frames] for 2d and  [batch, feats-dim, frames] for 1d.
        # Should keep the channel/plane is always in 1-dim of tensor (index-0 based).
        inplanes = 1 if self.convXd == 2 else inputs_dim

        self.resnet = ResNet(inplanes, **resnet_params)

        # It is just equal to Ceil function.
        resnet_output_dim = (
                                        inputs_dim + self.resnet.get_downsample_multiple() - 1) // self.resnet.get_downsample_multiple() \
                            * self.resnet.get_output_planes() if self.convXd == 2 else self.resnet.get_output_planes()

        # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.spk1_stats = LDEPooling(resnet_output_dim, c_num=pooling_params["num_head"])
            self.lang_stats = LDEPooling(resnet_output_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.spk1_stats = AttentiveStatisticsPooling(resnet_output_dim, hidden_size=pooling_params["hidden_size"],
                                                        context=pooling_params["context"], stddev=stddev)
            self.lang_stats = AttentiveStatisticsPooling(resnet_output_dim, hidden_size=pooling_params["hidden_size"],
                                                         context=pooling_params["context"], stddev=stddev)
        elif pooling == "multi-head":
            self.spk1_stats = MultiHeadAttentionPooling(resnet_output_dim, stddev=stddev, **pooling_params)
            self.lang_stats = MultiHeadAttentionPooling(resnet_output_dim, stddev=stddev, **pooling_params)
        elif pooling == "multi-resolution":
            self.spk1_stats = MultiResolutionMultiHeadAttentionPooling(resnet_output_dim, **pooling_params)
            self.lang_stats = MultiResolutionMultiHeadAttentionPooling(resnet_output_dim, **pooling_params)
        else:
            self.spk1_stats = StatisticsPooling(resnet_output_dim, stddev=stddev)
            self.lang_stats = StatisticsPooling(resnet_output_dim, stddev=stddev)

        self.spk1_fc1 = ReluBatchNormTdnnLayer(self.spk1_stats.get_output_dim(), resnet_params["planes"][3],
                                              **fc1_params) if fc1 else None
        self.lang_fc1 = ReluBatchNormTdnnLayer(self.lang_stats.get_output_dim(), resnet_params["planes"][3],
                                               **fc1_params) if fc1 else None

        if fc1:
            fc2_in_dim = resnet_params["planes"][3]
        else:
            fc2_in_dim = self.spk1_stats.get_output_dim()

        self.spk1_fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, resnet_params["planes"][3], **fc2_params)
        self.lang_fc2 = ReluBatchNormTdnnLayer(fc2_in_dim, resnet_params["planes"][3], **fc2_params)

        self.lang_output = nn.Linear(resnet_params["planes"][3], num_lang)
        self.MINE = MINE(512)

        self.discriminator1 = nn.Linear(256, 128)
        self.discriminator2 = nn.Linear(128, 2)

        self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None
        self.embd_dim = resnet_params["planes"][3]
        ## Do not need when extracting embedding.
#        self.lang_loss = nn.CrossEntropyLoss()

            # An example to using transform-learning without initializing loss.affine parameters
            # self.transform_keys = ["resnet", "stats", "fc1", "fc2"]

 #           self.transform_keys = ["resnet", "stats", "fc1", "fc2", "loss.weight"]

 #           if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
 #               self.rename_transform_keys = {"loss.affine.weight": "loss.weight"}

    @torch.jit.unused
    @utils.for_device_free
    def forward(self, x, alpha, x_len: torch.Tensor = torch.empty(0)):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        outputs = []
        x = self.cmvn_(x)

        x = self.auto(self.aug_dropout,
                      x)  # This auto function is equal to "x = layer(x) if layer is not None else x" for convenience.
        # [samples-index, frames-dim-index, frames-index] -> [samples-index, 1, frames-dim-index, frames-index]
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        # [samples-index, channel, frames-dim-index, frames-index] -> [samples-index, channel*frames-dim-index, frames-index]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) if self.convXd == 2 else x
        fspk = self.spk1_stats(x)
        flang = self.lang_stats(x.detach())
        lang6 = self.auto(self.lang_fc1, flang)
        lang = self.lang_fc2(lang6)
        lang_logit = self.lang_output(lang.view(-1, self.embd_dim))
        outputs.append(lang_logit)

        spk6 = self.auto(self.spk1_fc1, fspk)
        spk = self.spk1_fc2(spk6)
        spk = self.auto(self.tail_dropout, spk)
        outputs.append(spk6)
        outputs.append(spk)

        l_shuffle = torch.cat((lang[1:], lang[0].unsqueeze(0)), 0)
        ls = torch.cat((spk6.squeeze(), lang.squeeze().detach()), 1)
        ls_shuffle = torch.cat((spk6.squeeze(), l_shuffle.squeeze().detach()), 1)
        reverse_ls = GradReverse.apply(ls, alpha)
        reverse_ls_shuffle = GradReverse.apply(ls_shuffle, alpha)

        pos = -F.softplus(-self.MINE(reverse_ls))
        neg = F.softplus(-self.MINE(reverse_ls_shuffle)) + self.MINE(reverse_ls_shuffle)
        mi = neg.mean() - pos.mean()
        outputs.append(mi)

        reverse_emb = GradReverse.apply(spk.squeeze(), alpha)
        flogits = F.relu_(self.discriminator1(reverse_emb))
        flogits = self.discriminator2(flogits)
        outputs.append(flogits)

        return outputs

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, x):
        """
        x: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """
        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = self.cmvn_(x)
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) if self.convXd == 2 else x
        x = self.spk1_stats(x)

        if self.extracted_embedding == "far":
            assert self.spk1_fc1 is not None
            xvector = self.spk1_fc1.affine(x)
        elif self.extracted_embedding == "near_affine":
            x = self.auto(self.spk1_fc1, x)
            xvector = self.spk1_fc2.affine(x)
        elif self.extracted_embedding == "near":
            x = self.auto(self.spk1_fc1, x)
            xvector = self.spk1_fc2(x)
        else:
            raise TypeError("Expected far or near position, but got {}".format(self.extracted_embedding))

        return xvector

    def extract_embedding_jit(self, x: torch.Tensor, position: str = 'near') -> torch.Tensor:
        """
        x: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = self.cmvn_(x)
        # Tensor shape is not modified in libs.nnet.resnet.py for calling free, such as using this framework in cv.
        x = x.unsqueeze(1) if self.convXd == 2 else x
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) if self.convXd == 2 else x
        x = self.stats(x)

        if position == "far" and self.fc1 is not None:
            xvector = self.fc1.affine(x)
        elif position == "near_affine":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2.affine(x)
        elif position == "near":
            if self.fc1 is not None:
                x = self.fc1(x)
            xvector = self.fc2(x)

        else:
            raise TypeError("Expected far or near position, but got {}".format(position))

        return xvector

    @torch.jit.export
    def extract_embedding_whole(self, input: torch.Tensor, position: str = 'near', maxChunk: int = 4000,
                                isMatrix: bool = True):
        with torch.no_grad():
            if isMatrix:
                input = torch.unsqueeze(input, dim=0)
                input = input.transpose(1, 2)
            num_frames = input.shape[2]
            num_split = (num_frames + maxChunk - 1) // maxChunk
            split_size = num_frames // num_split
            offset = 0
            embedding_stats = torch.zeros(1, self.embd_dim, 1).to(input.device)
            for _ in range(0, num_split - 1):
                this_embedding = self.extract_embedding_jit(
                    input[:, :, offset:offset + split_size], position)
                offset += split_size
                embedding_stats += split_size * this_embedding

            last_embedding = self.extract_embedding_jit(
                input[:, :, offset:], position)

            embedding = (embedding_stats + (num_frames - offset)
                         * last_embedding) / num_frames
            return torch.squeeze(embedding.transpose(1, 2)).cpu()

    @torch.jit.export
    def embedding_dim(self) -> int:
        """ Export interface for c++ call, return embedding dim of the model
        """
        return self.embd_dim


class DSAN(TopVirtualNnet):
    def init(self, inputs_dim, num_targets, num_lang=106, num_dom=11, aug_dropout=0., tail_dropout=0., training=True,
             extracted_embedding="near",
             cmvn=False, cmvn_params={},
             resnet_params={}, pooling="statistics", pooling_params={}, fc1=True, fc1_params={}, fc2_params={},
             margin_loss=False, margin_loss_params={},
             use_step=False, step_params={}, transfer_from="softmax_loss", jit_compile=False,
             num_per_class=4, emb_size=256):

        self.feature_extractor = ResNetXvector(inputs_dim, num_targets, num_lang=num_lang, num_dom=num_dom, aug_dropout=aug_dropout,
                                               tail_dropout=tail_dropout, training=training, extracted_embedding=extracted_embedding,
                                               cmvn=cmvn, cmvn_params=cmvn_params, resnet_params=resnet_params, pooling=pooling,
                                               pooling_params=pooling_params, fc1=fc1, fc1_params=fc1_params, fc2_params=fc2_params,
                                               margin_loss=margin_loss, margin_loss_params=margin_loss_params, use_step=use_step,
                                               step_params=step_params, transfer_from=transfer_from, jit_compile=jit_compile)
        self.lang_loss = nn.CrossEntropyLoss()
        self.mmd_loss = mmd.MaximumMeanDiscrepancy()
        self.loss_d = torch.nn.CrossEntropyLoss()
        self.num_per_class = num_per_class
        self.emb_size = emb_size

        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(resnet_params["planes"][3], num_targets, **margin_loss_params)
                if self.use_step and self.step_params["margin_warm"]:
                    self.margin_warm = MarginWarm(**step_params["margin_warm_conf"])
            else:
                self.loss = SoftmaxLoss(resnet_params["planes"][3], num_targets)

        self.ge2e_loss = GE2ELoss(init_w=10.0, init_b=-5.0)

    @utils.for_device_free
    def forward(self, source_inputs, target_inputs, alpha):
        result = []
        slang_logit, sspk, sspk_logit, smi_lang, sflogits = self.feature_extractor(source_inputs, alpha)
        tlang_logit, tspk, tspk_logit, tmi_lang, tflogits = self.feature_extractor(target_inputs, alpha)

        lang_logit = torch.cat((slang_logit, tlang_logit), 0)
        mi_lang = 0.5*smi_lang + 0.5*tmi_lang
        flogits = torch.cat((sflogits, tflogits), 0)

        simmat_source, ssim_lbl = self.spk_mat(sspk_logit.squeeze())
        simmat_target, tsim_lbl = self.spk_mat(tspk_logit.squeeze())

        sdist, swc, sbc = gen_dist(simmat_source)
        tdist, twc, tbc = gen_dist(simmat_target)

        loss_mmd_wc, loss_mmd_bc = self.mmd_loss(swc, twc, sbc, tbc)
        loss_mmd = loss_mmd_wc + 2 * loss_mmd_bc

        result.append(lang_logit)
        result.append(sspk_logit)
        result.append(mi_lang)
        result.append(flogits)
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
    def get_loss(self, lang, spk, mi, flogits, tspk_loss, loss_mmd, targets, lang_labels, domains):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        spk_loss = self.loss(spk, targets)
        lang_loss = self.lang_loss(lang, lang_labels)
        mi_loss = mi - torch.log(torch.tensor(4.0))
        loss_d = self.loss_d(flogits, domains) - torch.log(torch.tensor(2.0))

        return spk_loss, lang_loss, mi_loss, loss_d, tspk_loss, loss_mmd

    @utils.for_device_free
    def get_closs(self, lang, spk, mi, flogits, tspk_loss, loss_mmd, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        """
        return self.loss(spk, targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

    @utils.for_device_free
    def extract_embedding(self, inputs):
        embedding = self.feature_extractor.extract_embedding(inputs)
        return embedding

    def get_warmR_T(self, T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end) / (T_i - 1) * (T_cur % T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch * epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"] * (
                                                1 + self.step_params["gamma"] * current_postion) ** (
                                        -self.step_params["alpha"]))
                lambda_m = 1 / (1 + lambda_factor)
                self.loss.step(lambda_m)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur * epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def step_iter(self, epoch, cur_step):
        # For iterabledataset
        if self.use_step:
            if self.step_params["margin_warm"]:
                offset_margin, lambda_m = self.margin_warm.step(cur_step)
                lambda_m = max(1e-3, lambda_m)
                self.loss.step(lambda_m, offset_margin)
            if self.step_params["m"]:
                lambda_factor = max(self.step_params["lambda_0"],
                                    self.step_params["lambda_b"] * (1 + self.step_params["gamma"] * cur_step) ** (
                                        -self.step_params["alpha"]))
                lambda_m = 1 / (1 + lambda_factor)
                self.loss.step(lambda_m)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = self.get_warmR_T(*self.step_params["T"], cur_step)

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(
                    *self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(
                    *self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]

    def load_transform_state_dict(self, state_dict):
        assert isinstance(self.transform_keys, list)
        assert isinstance(self.rename_transform_keys, dict)

        remaining = {utils.key_to_value(self.rename_transform_keys, k, False): v for k, v in state_dict.items() if
                     k.split('.')[0] \
                     in self.transform_keys or k in self.transform_keys}
        self.load_state_dict(remaining, strict=False)

#    def load_pretrained_module(self, initializer):
#        pretrained_dict = torch.load(initializer)
#        loss_dict = self.state_dict()
#        train_loss_dict = {k: v for k, v in pretrained_dict.items() if k in loss_dict}
#        loss_dict.update(train_loss_dict)
#        self.load_state_dict(loss_dict, strict=False)

#        model_dict = self.state_dict()
#        layer_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#        model_dict.update(layer_dict)
#        self.load_state_dict(model_dict, strict=False)
#        loss_dict.update()
#        return self

    def load_pretrained_module(self, initializer):
        pretrained_dict = torch.load(initializer)
        loss_dict = self.state_dict()
        train_loss_dict = {k: v for k, v in pretrained_dict.items() if k in loss_dict}
        loss_dict.update(train_loss_dict)
        self.load_state_dict(loss_dict, strict=False)

        model_dict = self.feature_extractor.state_dict()
        layer_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(layer_dict)
        self.feature_extractor.load_state_dict(model_dict, strict=False)
        loss_dict.update()
        return self