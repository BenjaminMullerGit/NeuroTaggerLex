
from torch.optim import Adam



def generate_optimizer(opt, lr, params, eps,gamma,betas):
    assert opt == "Adam"
    params = filter(lambda param: param.requires_grad, params)
    if opt == 'Adam':
        return Adam(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)