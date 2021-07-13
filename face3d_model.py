import sys
sys.path.append('../3DDFA_V2')
import os
from FaceBoxes import models
from utils.functions import draw_landmarks 
import cv2
import torch
import torch.nn as nn
import yaml
import numpy as np
from TDDFA import TDDFA

cfg_file = 'configs/mb1_120x120.yml'
cfg = yaml.load(open(cfg_file), Loader=yaml.SafeLoader)
test_img = 'data/emma.jpg'

gt_box = [1699.8129, 278.4989, 2057.769, 762.43463, 0.9999492]

def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[-1]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[..., :trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[..., trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[..., trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

def original_result(display = False):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    tddfa = TDDFA(gpu_mode='gpu', **cfg)
    img = cv2.imread(test_img)
    param_lst, roi_box_lst = tddfa(img, [gt_box])
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag= False)
    if display:
        draw_landmarks(img, ver_lst, show_flag= True, dense_flag=False, wfp=None)
    return ver_lst[0].T

class ComposeModel(nn.Module):
    def __init__(self):
        super(ComposeModel, self).__init__()
        tddfa = TDDFA(**cfg)
        self.model = tddfa.model
        self.size = tddfa.size
        self.transform = tddfa.transform
        self.param_std = torch.from_numpy(tddfa.param_std)
        self.param_mean = torch.from_numpy(tddfa.param_mean)
        self.u_base = torch.from_numpy(tddfa.bfm.u_base)
        self.w_shp_base = torch.from_numpy(tddfa.bfm.w_shp_base)
        self.w_exp_base = torch.from_numpy(tddfa.bfm.w_exp_base)
        self.eval()

    def preprocess(self, roi_img):
        img = cv2.resize(roi_img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return self.transform(img)

    def forward(self, norm_img):
        param = self.model(norm_img)
        param = param * self.param_std + self.param_mean
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        vertex = torch.transpose((self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).reshape(-1, 3), 0, 1)
        pts3d =  torch.transpose(R@vertex + offset, 0, 1)
        import ipdb;ipdb.set_trace()
        return pts3d
    
    def postprocess(self, pts3d):
        # pts3d[0, :] -= 1  # for Python compatibility
        # pts3d[2, :] -= 1
        # pts3d[1, :] = self.size - pts3d[1, :]
        # pts3d[2, :] -= np.min(pts3d[2, :])
        pts3d[:, 0] -= 1
        pts3d[:, 2] -= 1
        pts3d[:, 1] = self.size - pts3d[:, 1]
        pts3d[:, 2] -= np.min(pts3d[:, 2])
        return np.array(pts3d, dtype=np.float32)

def test_compose_model():
    face = cv2.imread('crop_face.png')
    model = ComposeModel()
    with torch.no_grad():
        input_tensor = model.preprocess(face)
        points = model(input_tensor[None, :, :, :])
    pts = model.postprocess(points.numpy())
    resized_face = cv2.resize(face, (120, 120))
    for pt in pts[:, :2].astype(np.int32):
        print(pt)
        cv2.circle(resized_face, tuple(pt), 1, (0, 0 , 255), thickness=-1)
    cv2.imshow('img', resized_face)
    cv2.waitKey(0)

def ConvertTOOnnx():
    model = ComposeModel()
    model.eval()
    dummy_input = torch.randn(1, 3, model.size, model.size)
    out_onnx = './models/face3d.onnx'
    torch.onnx.export(
        model,
        (dummy_input,),
        out_onnx,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True)
    

if __name__ == '__main__':
    # original_result(True)
    test_compose_model()
    # ConvertTOOnnx()