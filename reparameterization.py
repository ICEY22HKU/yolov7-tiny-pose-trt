# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml
import argparse

parser = argparse.ArgumentParser(description = "reparameterization for yolov7 tiny")
parser.add_argument("--weights", type= str, default='yolov7-tiny_training.pt', help='initial weights path')
parser.add_argument("--classes", type=int, default=80, help='classs')
parser.add_argument("--deploy_yaml", type=str, default='cfg/deploy/yolov7-tiny.yaml', help='path to the deploy yaml file')
args = parser.parse_args()

device = select_device('0', batch_size=1)
# model trained by cfg/training/*.yaml
ckpt = torch.load(args.weights, map_location=device)
# reparameterized model in cfg/deploy/*.yaml
model = Model(args.deploy_yaml, ch=3, nc=args.classes).to(device)

with open(args.deploy_yaml) as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'][0]) // 2

# copy intersect weights
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
model.load_state_dict(intersect_state_dict, strict=False)
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

# reparametrized YOLOR
for i in range((model.nc+5)*anchors):
    model.state_dict()['model.77.m.0.weight'].data[i, :, :, :] *= state_dict['model.77.im.0.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.77.m.1.weight'].data[i, :, :, :] *= state_dict['model.77.im.1.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.77.m.2.weight'].data[i, :, :, :] *= state_dict['model.77.im.2.implicit'].data[:, i, : :].squeeze()
model.state_dict()['model.77.m.0.bias'].data += state_dict['model.77.m.0.weight'].mul(state_dict['model.77.ia.0.implicit']).sum(1).squeeze()
model.state_dict()['model.77.m.1.bias'].data += state_dict['model.77.m.1.weight'].mul(state_dict['model.77.ia.1.implicit']).sum(1).squeeze()
model.state_dict()['model.77.m.2.bias'].data += state_dict['model.77.m.2.weight'].mul(state_dict['model.77.ia.2.implicit']).sum(1).squeeze()
model.state_dict()['model.77.m.0.bias'].data *= state_dict['model.77.im.0.implicit'].data.squeeze()
model.state_dict()['model.77.m.1.bias'].data *= state_dict['model.77.im.1.implicit'].data.squeeze()
model.state_dict()['model.77.m.2.bias'].data *= state_dict['model.77.im.2.implicit'].data.squeeze()

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
torch.save(ckpt, 'yolov7-tiny.pt')
