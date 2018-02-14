import os
os.environ["CUDA_VISIBLE_DEVICES"]=1

import torch

import paragraphvec.export_vectors as ev

input_data = '/home/dgopstein/opt/src/paragraph-vectors/data/nginx_source.csv'
saved_model = '/home/dgopstein/opt/src/paragraph-vectors/models/nginx_source_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.93_loss.0.829367.pth.tar'

#checkpoint = torch.load(saved_model, map_location={'cuda:0':'cuda:1'})
#
#model_state_dict = checkpoint['model_state_dict']
#model_state_dict.keys()


## Export the generated model to CSV
ev.start(input_data, saved_model)

'nginx_source_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.62_loss.0.677806.csv'
