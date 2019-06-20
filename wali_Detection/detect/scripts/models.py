from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        """
        :param anchors: [(23,27), (37,58), (81,82)]
        :param num_classes: 80
        :param img_dim: 416
        """
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5

        self.mse_loss = nn.MSELoss()  # mean square
        self.bce_loss = nn.BCELoss()  # binary cross entropy
        self.obj_scale = 1
        self.noobj_scale = 100  # noobj
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        """
        :param grid_size: usually is end fm size, h = w, 13,26
        :param cuda:
        :return:
        """
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size  # 416/13=32 grid_w
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  # 13x13
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        # scale anchors to fm_size (anchor/stride), ori anchors: [(23,27), (37,58), (81,82)] / 32
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        # w, h --> (1,3,1,1)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        """
        :param x: fm of previous layer, BxCxHxW
        :param targets:
        :param img_dim:
        :return:
        """
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)  # batch
        grid_size = x.size(2)  # exactly is fm size, 13, 26

        # B C H W
        # B,3,85,H,W -> B,3,H,W,85
        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 5,
                            grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs, total bboxes: 3xHxW = 3x13x13
        x = torch.sigmoid(prediction[..., 0])  # Center x   wrt grid h,w
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width    wrt ori_img h,w
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf      B,3,H,W
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. B,3,H,W,80

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:  # assign new grid, update grid_size, stride, scaled_anchors
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # x,y,w,h, B,3,H,W,4
        # pred_offset(x,y) + begin(x,y)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        # scale pred_bbox(w,h) to anchors we set
        # cus tw = log(gw/anchor_w), gw groundtruth w
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat((
            pred_boxes.view(num_samples, -1, 4) * self.stride,  # B, 3x13x13, 4
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, self.num_classes),
        ), -1)  # B, 3x13x13, 85

        if targets is None:
            return output, 0
        else:  # if train, obj_mask, noobj_mask from target
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,  # B,3,H,W,4 [center_x, center_y, w, h]
                pred_cls=pred_cls,  # B,3,H,W,80
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # coordinates
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # obj_mask: best_anchor & best grid
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            # grid confidence
            # obj_mask, GT grid
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # tconf = obj_mask.float()
            # noobj_mask, 1.grid has no bbox; 2.grid has bbox < ignore_thres
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # obj_scale=1, noobj_scale=100, as noobj >> obj
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # class
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])  # one-hot
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics, mean
            cls_acc = 100 * class_mask[obj_mask].mean()  # class_mask [0,1]
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            # recall of iou50,75
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # pop net hyperparams
    output_filters = [int(hyperparams["channels"])]  # 3
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                name="conv_{}".format(module_i),  # f-strings format
                module=nn.Conv2d(
                    in_channels=output_filters[-1],  # out filters of previous module
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                )
            )
            if bn:
                modules.add_module("batch_norm_{}".format(module_i),
                                   nn.BatchNorm2d(num_features=filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_{}".format(module_i),
                                   nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:  # if stride=1
                modules.add_module("_debug_padding_{}".format(module_i),
                                   nn.ZeroPad2d((0, 1, 0, 1)))  # lrtb
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{}".format(module_i), maxpool)

        elif module_def["type"] == "upsample":  # 19
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")  # self-defined
            modules.add_module("upsample_{}".format(module_i), upsample)

        elif module_def["type"] == "route":  # 17, 20
            # route is a placeholder, just to get filters, concat layers
            # 17, [-4]      layers=14,    filters=512
            # 20, [-1, 8]   layers=19,8   filters=128+256
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module("route_{}".format(module_i), EmptyLayer())  # self-defined

        elif module_def["type"] == "shortcut":  # yolov3 has, tiny doesn't
            # similar as route
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module("shortcut_{}".format(module_i), EmptyLayer())  # self-defined

        elif module_def["type"] == "yolo":  # 16, 23
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            # [(10,14), (23,27), (37,58), (81,82), (135,169), (344,319)]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # 16 [3,4,5], 23 [1,2,3]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])  # 80
            img_size = int(hyperparams["height"])  # 416
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(module_i), yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)  # each modules is nn.Sequential()

        # filters store in [output_filters] for 'route' and 'shortcut'
        output_filters.append(filters)

    return hyperparams, module_list


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        # config_path: model .cfg file
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)  # list [hyper params, layer params]
        # build pytroch net layer using model_defs
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # nn.Sequential(), class YOLOLayer has attr "self.metrics"
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []  # store middle results for route, shortcut
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                # concat tensors in []
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                # previous + shortcut
                x = layer_outputs[-1] + layer_outputs[int(module_def["from"])]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)  # yolo layer forward
                loss += layer_loss  # 16,23
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))  # 16, 13x13x255; 23, 26x26x255
        return yolo_outputs if targets is None else (loss, yolo_outputs)  # if no targets, means test

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
