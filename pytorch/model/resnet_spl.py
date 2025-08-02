# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-28)

import sys
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, "subtools/pytorch")

import libs.support.utils as utils
from libs.nnet import *

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

        self.tail_dropout = torch.nn.Dropout2d(p=tail_dropout) if tail_dropout > 0 else None
        self.embd_dim = resnet_params["planes"][3]
        ## Do not need when extracting embedding.
        self.lang_loss = nn.CrossEntropyLoss()
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(resnet_params["planes"][3], num_targets, **margin_loss_params)
                if self.use_step and self.step_params["margin_warm"]:
                    self.margin_warm = MarginWarm(**step_params["margin_warm_conf"])
            else:
                self.loss = SoftmaxLoss(resnet_params["planes"][3], num_targets)
            # An example to using transform-learning without initializing loss.affine parameters
            # self.transform_keys = ["resnet", "stats", "fc1", "fc2"]

            self.transform_keys = ["resnet", "stats", "fc1", "fc2", "loss.weight"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight": "loss.weight"}

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

        return outputs

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
        return self.loss(spk, targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        return self.loss.get_posterior()

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

    def load_pretrained_module(self, initializer):
        pretrained_dict = torch.load(initializer)
#        loss_dict = self.state_dict()
#        train_loss_dict = {k: v for k, v in pretrained_dict.items() if k in loss_dict}
#        loss_dict.update(train_loss_dict)
#        self.load_state_dict(loss_dict, strict=False)

        model_dict = self.state_dict()
        layer_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(layer_dict)
        self.load_state_dict(model_dict, strict=False)
#        loss_dict.update()
        return self