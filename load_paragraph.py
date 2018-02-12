import torch

saved_model = 'nginx_source_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.62_loss.0.677806.pth.tar'

checkpoint = torch.load(saved_model)
net = torch.load(checkpoint['model'])
