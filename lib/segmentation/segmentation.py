"""
Segmentation:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'label': [batch_size, 1] <- [batch_size, c, h, w]}
"""

import numpy as np
from utils.image import get_segmentation_pair, tensor_vstack

def get_segmentation_test_batch(segdb, config):
    """
    return a dict of train batch
    :param segdb: ['image', 'flipped']
    :param config: the config setting
    :return: data, label, im_info
    """
    imgs, seg_cls_gts, segdb = get_segmentation_image(segdb, config)
    im_array = imgs
    im_info = [np.array([segdb[i]['im_info']], dtype=np.float32) for i in xrange(len(segdb))]

    data = [{'data': im_array[i],
            'im_info': im_info[i]} for i in xrange(len(segdb))]
    label = [{'label':seg_cls_gts[i]} for i in xrange(len(segdb))]

    return data, label, im_info

def get_segmentation_train_batch(segdb, config):
    """
    return a dict of train batch
    :param segdb: ['image', 'flipped']
    :param config: the config setting
    :return: data, label, im_info
    """
    # assert len(segdb) == 1, 'Single batch only'
    assert len(segdb) == 1, 'Single batch only'

    mvs, ref_prev_imgs, ref_next_imgs, eq_flags, seg_cls_gts, segdb = get_segmentation_pair(segdb, config)
    mv_array = mvs[0]
    ref_prev_im_array = ref_prev_imgs[0]
    ref_next_im_array = ref_next_imgs[0]
    eq_flag_array = np.array([eq_flags[0],], dtype=np.float32)
    seg_cls_gt = seg_cls_gts[0]

    im_info = np.array([segdb[0]['im_info']], dtype=np.float32)

    data = {'data': mv_array,
            'data_ref_prev': ref_prev_im_array,
            'data_ref_next': ref_next_im_array,
            'eq_flag': eq_flag_array,
            'im_info': im_info}
    label = {'label': seg_cls_gt}

    return data, label

