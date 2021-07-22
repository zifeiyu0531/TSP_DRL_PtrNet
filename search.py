import torch
import os
from actor import PtrNet1


def sampling(cfg, env, test_input):
    test_inputs = test_input.repeat(cfg.batch, 1, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    act_model = PtrNet1(cfg)
    if os.path.exists(cfg.act_model_path):
        act_model.load_state_dict(torch.load(cfg.act_model_path, map_location=device))
    else:
        print('specify pretrained model path')
    act_model = act_model.to(device)
    pred_tours, _ = act_model(test_inputs, device)
    l_batch, _, _, _, _ = env.stack_l_fast(test_inputs, pred_tours)
    index_lmin = torch.argmin(l_batch)
    best_tour = pred_tours[index_lmin]
    return best_tour
