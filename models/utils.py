# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

_MODELS = {}


def register_model(cls=None, *, name=None):
    '''A decorator for registering model classes.'''

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(
                'Already registered model with name: %s' % local_name)
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config):
    model_name = config.name
    score_model = get_model(model_name)(config)
    return score_model


def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(config, sde, model, train=False):
    model_fn = get_model_fn(model, train=train)

    def score_fn(u, t):
        score = model_fn(u.type(torch.float32), t.type(torch.float32))
        noise_multiplier = sde.noise_multiplier(t).type(torch.float32)

        if config.mixed_score:
            if sde.is_augmented:
                _, z = torch.chunk(u, 2, dim=1)
                ones = torch.ones_like(z, device=config.device)
                var_zz = (sde.var(t, 0. * ones, (sde.gamma / sde.m_inv) * ones)[2]).type(torch.float32)
                return - z / var_zz + score * noise_multiplier
            else:
                ones = torch.ones_like(u, device=config.device)
                var = (sde.var(t, ones)[0]).type(torch.float32)
                return -u / var + score * noise_multiplier
        else:
            return noise_multiplier * score
    return score_fn

def get_x0_prediction(config, sde, model, train=False):
    def x0_prediction(u, t):
        score_fn = get_score_fn(config, sde, model, train)
        noise_multiplier = sde.noise_multiplier(t).type(torch.float32)
        eps_prediction = score_fn(u, t) / noise_multiplier

        beta_int = sde.beta_int_fn(t)
        if sde.is_augmented:
            coeff = torch.exp(-2. * beta_int * sde.g)
            g = - beta_int * coeff
            f = (2. * sde.g * beta_int + 1.) * coeff
            var = sde.var(t)
            cholesky11 = (torch.sqrt(var[0]))
            cholesky21 = (var[1] / cholesky11)
            cholesky22 = (torch.sqrt(var[2] - cholesky21 ** 2.))

            x, v = torch.chunk(u, 2, dim=1)
            x0 = (v - cholesky22 * eps_prediction - cholesky21 * x / cholesky11) / (g - f * cholesky21 / cholesky11)
            return x0
        else:
            coeff = torch.exp(-0.5 * beta_int)
            return (u - (1. - coeff ** 2.)  * eps_prediction) / coeff
    return x0_prediction
