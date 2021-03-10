import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None, use_gpu=True):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = torch.Tensor([0.5, 1, 2])
        if scales is None:
            self.scales = torch.Tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        self.use_gpu = use_gpu
        
        if self.use_gpu and torch.cuda.is_available():
            self.scales = self.scales.cuda()
            self.ratios = self.ratios.cuda()

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = torch.Tensor([image_shape[0], image_shape[1]])

        if self.use_gpu and torch.cuda.is_available():
            image_shape = image_shape.cuda()

        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = []

        for idx in range(len(self.pyramid_levels)):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales, use_gpu=self.use_gpu)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors, use_gpu=self.use_gpu)
            all_anchors.append(shifted_anchors)

        all_anchors = torch.cat(all_anchors).unsqueeze(dim=0)

        return all_anchors

    def switch_to_cpu(self):
        self.use_gpu = False
        self.scales = self.scales.cpu()
        self.ratios = self.ratios.cpu()
    
    def switch_to_gpu(self):
        self.use_gpu = True
        self.scales = self.scales.cuda()
        self.ratios = self.ratios.cuda()


def generate_anchors(base_size, ratios, scales, use_gpu):
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = torch.zeros((num_anchors, 4))
    if use_gpu and torch.cuda.is_available():
        anchors = anchors.cuda()

    # scale base_size
    anchors[:, 2:] = base_size * scales.repeat(2, len(ratios)).t()

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = torch.sqrt(areas / ratios.repeat(len(scales)))
    anchors[:, 3] = anchors[:, 2] * ratios.repeat(len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= (anchors[:, 2]*0.5).repeat(2, 1).t()
    anchors[:, 1::2] -= (anchors[:, 3]*0.5).repeat(2, 1).t()

    return anchors


def shift(shape, stride, anchors, use_gpu):
    shift_x = (torch.arange(0, shape[1]) + 0.5) * stride
    shift_y = (torch.arange(0, shape[0]) + 0.5) * stride

    if use_gpu and torch.cuda.is_available():
        shift_x = shift_x.cuda()
        shift_y = shift_y.cuda()

    # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    len_shift_x, len_shift_y = len(shift_x), len(shift_y)
    shift_x, shift_y = shift_x.repeat(len_shift_y, 1), shift_y.repeat(len_shift_x, 1).t()

    # shifts = np.vstack((
    #     shift_x.ravel(), shift_y.ravel(),
    #     shift_x.ravel(), shift_y.ravel()
    # )).transpose()
    shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)]).t()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape(1, A, 4) + shifts.reshape(1, K, 4).permute(1, 0, 2))
    all_anchors = all_anchors.reshape(K * A, 4)

    return all_anchors

