import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_net(net, loader, device,
             mode='amplitude+phase', middleout=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # loader means dataloader here
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            amplitude_input = batch['amplitude_input']  # shape: BCHW
            phase_input = batch['phase_input']
            amplitude_target = batch['amplitude_target']
            phase_target = batch['phase_target']

            input_data = torch.cat((amplitude_input, phase_input), 1)
            if mode == 'amplitude':
                target_data = amplitude_target
            elif mode == 'phase':
                target_data = phase_target
            else:
                target_data = torch.cat((amplitude_target, phase_target), 1)

            input_data = input_data.to(device=device, dtype=torch.float32)
            target_data = target_data.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if middleout:
                    pred_data, _ = net(input_data)
                else:
                    pred_data = net(input_data)

            tot += F.mse_loss(pred_data, target_data).item()
            pbar.update(input_data.shape[0])

    net.train()
    return tot / n_val
