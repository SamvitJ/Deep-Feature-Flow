# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
from PIL import Image
import numpy as np
import pickle

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/dff_deeplab/cfgs/dff_deeplab_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_segment, Predictor
from symbols import *
from utils.load_model import load_param_multi
from utils.show_boxes import show_boxes, draw_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

ref_img_prefix = 'frankfurt_000000_000294'
ref_pred_prefix = 'seg_frankfurt_000000_012000'

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deep Feature Flow demo')
    parser.add_argument('-i', '--interval', type=int, default=1)
    parser.add_argument('-e', '--num_ex', type=int, default=10)
    parser.add_argument('--gt', dest='has_gt', action='store_true')
    parser.add_argument('--no_gt', dest='has_gt', action='store_false')
    parser.add_argument('--diff', dest='diff', action='store_true')
    parser.set_defaults(has_gt=True)
    parser.set_defaults(diff=False)
    args = parser.parse_args()
    return args

args = parse_args()

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.true_divide(np.diag(hist), (hist.sum(1) + hist.sum(0) - np.diag(hist)))

def get_label_if_available(label_files, im_filename):
    for lb_file in label_files:
        _, lb_filename = os.path.split(lb_file)
        lb_filename = lb_filename[:len(ref_img_prefix)]
        if im_filename.startswith(lb_filename):
            print 'label {}'.format(lb_filename)
            return lb_file
    return None

def getpallete(num_cls):
    """
    this function is to get the colormap for visualizing the segmentation mask
    :param num_cls: the number of visulized class
    :return: the pallete
    """
    n = num_cls
    pallete_raw = np.zeros((n, 3)).astype('uint8')
    pallete = np.zeros((n, 3)).astype('uint8')

    pallete_raw[6, :] =  [111,  74,   0]
    pallete_raw[7, :] =  [ 81,   0,  81]
    pallete_raw[8, :] =  [128,  64, 128]
    pallete_raw[9, :] =  [244,  35, 232]
    pallete_raw[10, :] =  [250, 170, 160]
    pallete_raw[11, :] = [230, 150, 140]
    pallete_raw[12, :] = [ 70,  70,  70]
    pallete_raw[13, :] = [102, 102, 156]
    pallete_raw[14, :] = [190, 153, 153]
    pallete_raw[15, :] = [180, 165, 180]
    pallete_raw[16, :] = [150, 100, 100]
    pallete_raw[17, :] = [150, 120,  90]
    pallete_raw[18, :] = [153, 153, 153]
    pallete_raw[19, :] = [153, 153, 153]
    pallete_raw[20, :] = [250, 170,  30]
    pallete_raw[21, :] = [220, 220,   0]
    pallete_raw[22, :] = [107, 142,  35]
    pallete_raw[23, :] = [152, 251, 152]
    pallete_raw[24, :] = [ 70, 130, 180]
    pallete_raw[25, :] = [220,  20,  60]
    pallete_raw[26, :] = [255,   0,   0]
    pallete_raw[27, :] = [  0,   0, 142]
    pallete_raw[28, :] = [  0,   0,  70]
    pallete_raw[29, :] = [  0,  60, 100]
    pallete_raw[30, :] = [  0,   0,  90]
    pallete_raw[31, :] = [  0,   0, 110]
    pallete_raw[32, :] = [  0,  80, 100]
    pallete_raw[33, :] = [  0,   0, 230]
    pallete_raw[34, :] = [119,  11,  32]

    train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    for i in range(len(train2regular)):
        pallete[i, :] = pallete_raw[train2regular[i]+1, :]

    pallete = pallete.reshape(-1)

    return pallete

def compare_imgs(curr_img, next_img):

    # print curr_img
    # print 'curr type', np.ndarray.dtype(curr_img)
    # print 'next type', np.ndarray.dtype(next_img)

    diff = next_img - curr_img
    # print diff
    # print 'max', np.amax(curr_img), np.amax(next_img), np.amax(diff)
    # print 'min', np.amin(curr_img), np.amin(next_img), np.amin(diff)
    # print 'avg', np.mean(curr_img), np.mean(next_img), np.mean(abs(diff))

    print np.shape(diff)
    flat_diff = np.ravel(abs(diff))

    def diff_greater(k, delta):
        deltas = [delta[0][0], delta[0][1], delta[0][2]]
        deltas = [np.ravel(d) for d in deltas]
        fracs = [len(np.where(d > k)[0]) / (1. * np.size(d)) for d in deltas]
        # print fracs
        return max(fracs)

    # def diff_greater(k):
    #     return len(np.where(flat_diff > k)[0]) / (1. * np.size(flat_diff))

    # print '0-norm', np.linalg.norm(flat_diff, ord=0) / (1. * np.size(flat_diff))
    diff_0  = diff_greater(1e-5, abs(diff))
    diff_5  = diff_greater(5, abs(diff))
    diff_10 = diff_greater(10, abs(diff))

    print 'diff >  0', diff_0
    print 'diff >  5', diff_5
    print 'diff > 10', diff_10

    return diff_0, diff_5, diff_10

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_flownet_deeplab'
    model1 = '/../model/rfcn_dff_flownet_vid'
    model2 = '/../model/deeplab_dcn_cityscapes'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    key_sym = sym_instance.get_key_test_symbol(config)
    next_key_sym = sym_instance.get_key_test_symbol(config)
    cur_sym = sym_instance.get_cur_test_symbol(config)

    # settings
    num_classes = 19
    snip_len = 30
    has_gt = args.has_gt
    interv = args.interval
    num_ex = args.num_ex
    comp_diff = args.diff

    # load demo data
    cstr = 'c' + str(interv)
    if has_gt:
        image_names = sorted(glob.glob('/city/leftImg8bit_sequence/val/frankfurt/*.png'))
        image_names = image_names[: snip_len * num_ex]
        label_files = sorted(glob.glob(cur_path + '/../demo/cityscapes_frankfurt_labels_all/*.png'))
        label_files = label_files[: num_ex]
    else:
        image_names = sorted(glob.glob(cur_path + '/../demo/cityscapes_frankfurt/*.png'))
        label_files = sorted(glob.glob(cur_path + '/../demo/cityscapes_frankfurt_preds/*.png'))
    output_dir = cur_path + '/../demo/deeplab_dff/'
    mv_file = '/city/leftImg8bit_sequence/val/frankfurt-all.pkl'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    key_frame_interval = interv

    # test params
    time = 0
    count = 0
    hist = np.zeros((num_classes, num_classes))
    lb_idx = 0

    print 'num snippets', (len(image_names) / snip_len)

    mvs = pickle.load(open(mv_file, 'rb'))
    mvs = np.transpose(mvs, (0, 3, 1, 2))
    print "mvs.shape %s" % (mvs.shape,)

    diff_0 = diff_5 = diff_10 = 0.
    for snip_idx in range(len(image_names) / snip_len):

        label_idx = 19
        offset = snip_idx % interv # rotate offsets in [0, interv - 1]
        start_pos = label_idx - offset
        snip_names = image_names[snip_idx * snip_len: (snip_idx + 1) * snip_len]
        snip_names = snip_names[start_pos: start_pos + interv + 1]

        #
        data = []
        mv_tensor = None

        print '\n\nsnippet', snip_idx, 'offset', offset
        for idx, im_name in enumerate(snip_names):
            assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
            im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            mv_tensor = np.expand_dims(mvs[(snip_idx * snip_len) + start_pos + idx], axis=0) / 16.
            data.append({
                'data': im_tensor,
                'im_info': im_info,
                'm_vec': mv_tensor,
                'feat_forw': np.zeros((1,config.network.DFF_FEAT_DIM,1,1)),
                'feat_back': np.zeros((1,config.network.DFF_FEAT_DIM,1,1)),
            })

        if comp_diff:
            for i in range(interv):
                print '\n', snip_names[i+1]
                frame_sym = mx.sym.Variable(name="frame")
                m_vec_sym = mx.sym.Variable(name="m_vec")

                m_vec_grid = mx.sym.GridGenerator(data=m_vec_sym, transform_type='warp', name='m_vec_grid')
                frame_warp = mx.sym.BilinearSampler(data=frame_sym, grid=m_vec_grid, name='warping_feat')

                frame_data = mx.nd.array(data[i]['data'])

                m_vec_data = data[i+1]['m_vec'] * 16.
                m_vec_data = np.repeat(m_vec_data, 16, axis=2)
                m_vec_data = np.repeat(m_vec_data, 16, axis=3)
                print np.shape(m_vec_data)
                # print m_vec_data
                m_vec_data = mx.nd.array(m_vec_data)
                m_vec_data = mx.ndarray.negative(m_vec_data)

                f_exec = frame_warp.bind(ctx=mx.gpu(),
                    args={"frame": frame_data, "m_vec": m_vec_data},
                    group2ctx={"frame": mx.gpu(), "m_vec": mx.cpu()})
                f_exec.forward()

                frame_i_warp = np.array(f_exec.outputs[0].asnumpy())

                print 'deltas'
                # diffs = compare_imgs(data[i]['data'], data[i+1]['data'])
                # diff_0  += diffs[0]
                # diff_5  += diffs[1]
                # diff_10 += diffs[2]
                # --interval 5 --num_ex 2 (total: 10)
                # avgs: 0.875080553691 0.354579051336 0.249108441671 (all channels)
                # avgs: 0.880920505524 0.358328533173 0.254517412186
                # --interval 5 --num_ex 10 (total: 50)
                # avgs: 0.856089115143 0.342332496643 0.227059316635
                # --interval 5 --num_ex 50 (total: 250)
                # avgs: 0.889601413727 0.423051366806 0.287254112244

                print 'residuals'
                diffs = compare_imgs(frame_i_warp, data[i+1]['data'])
                diff_0  += diffs[0]
                diff_5  += diffs[1]
                diff_10 += diffs[2]
                # --interval 5 --num_ex 2 (total: 10)
                # avgs: 0.859159898758 0.215072154999 0.0927718003591 (all channels)
                # avgs: 0.868176364899 0.221829843521 0.0966950416565
                # --interval 5 --num_ex 10 (total: 50)
                # avgs: 0.854715948105 0.249392948151 0.116685094833
                # --interval 5 --num_ex 50 (total: 250)
                # avgs: 0.882048036575 0.311128740311 0.153696537018
                # --interval 5 --num_ex 100 (total: 500)
                # avgs: 0.879370404243 0.296190703392 0.145828608513
                diff = abs(data[i+1]['data'] - frame_i_warp)

                x_inc = 512
                y_inc = 256
                for x in range(0, 2048, x_inc):
                    for y in range(0, 1024, y_inc):
                        diff_frag = diff[:, :, y : y + y_inc, x : x + x_inc]
                        print 'norms %4d %4d %6d %5.1d' % (x, y, np.linalg.norm(np.ravel(diff_frag), 2),
                            np.average(diff_frag))

                diff = np.transpose(np.uint8(np.squeeze(diff)), (1, 2, 0))
                # print diff
                print np.shape(diff)
                img = Image.fromarray(diff)
                _, im_filename = os.path.split(snip_names[i+1])
                img.save(output_dir + '/diff_' + im_filename)
            continue

        # get predictor
        data_names = ['data', 'm_vec', 'feat_forw', 'feat_back']
        label_names = []
        data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
        provide_label = [None for i in xrange(len(data))]
        # models: rfcn_dff_flownet_vid, deeplab_cityscapes
        arg_params, aux_params = load_param_multi(cur_path + model1, cur_path + model2, 0, process=True)
        key_predictor = Predictor(key_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        next_key_predictor = Predictor(next_key_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        cur_predictor = Predictor(cur_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        nms = gpu_nms_wrapper(config.TEST.NMS, 0)

        # warm up
        for j in xrange(2):
            data_batch = mx.io.DataBatch(data=[data[j]], label=[], pad=0, index=0,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[j])]],
                                         provide_label=[None])

            # scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            if j % key_frame_interval == 0:
                # scores, boxes, data_dict, feat = im_detect(key_predictor, data_batch, data_names, scales, config)
                output_all, feat = im_segment(key_predictor, data_batch)
                output_all = [mx.ndarray.argmax(output['croped_score_output'], axis=1).asnumpy() for output in output_all]

            else:
                data_batch.data[0][-2] = feat
                data_batch.provide_data[0][-2] = ('feat_forw', feat.shape)
                data_batch.data[0][-1] = feat
                data_batch.provide_data[0][-1] = ('feat_back', feat.shape)
                # scores, boxes, data_dict, _ = im_detect(cur_predictor, data_batch, data_names, scales, config)
                output_all, _ = im_segment(cur_predictor, data_batch)
                output_all = [mx.ndarray.argmax(output['croped_score_output'], axis=1).asnumpy() for output in output_all]

        print "warmup done"
        # test
        for idx, im_name in enumerate(snip_names[:key_frame_interval]):
            data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                         provide_label=[None])

            # load next keyframe data
            if idx % key_frame_interval == 0:
                assert (idx + key_frame_interval < len(snip_names))
                next_idx = idx + key_frame_interval
                data_batch_next = mx.io.DataBatch(data=[data[next_idx]], label=[], pad=0, index=next_idx,
                                                  provide_data=[[(k, v.shape) for k, v in zip(data_names, data[next_idx])]],
                                                  provide_label=[None])

            # scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            tic()
            if idx % key_frame_interval == 0:
                print '\nframe {} (key)     next {}'.format(idx, next_idx)
                # scores, boxes, data_dict, feat = im_detect(key_predictor, data_batch, data_names, scales, config)
                output_all, feat = im_segment(key_predictor, data_batch)
                output_all = [mx.ndarray.argmax(output['croped_score_output'], axis=1).asnumpy() for output in output_all]

                _, feat_next = im_segment(next_key_predictor, data_batch_next)

                forw_warp = [feat]
                for i in range(key_frame_interval):
                    feat_sym = mx.sym.Variable(name="feat")
                    m_vec_sym = mx.sym.Variable(name="m_vec")

                    m_vec_grid = mx.sym.GridGenerator(data=m_vec_sym, transform_type='warp', name='m_vec_grid')
                    feat_warp = mx.sym.BilinearSampler(data=feat_sym, grid=m_vec_grid, name='warping_feat')

                    m_vec_data = mx.ndarray.negative(data[idx + 1 + i][1])
                    f_exec = feat_warp.bind(ctx=mx.gpu(),
                        args={"feat": forw_warp[-1], "m_vec": m_vec_data},
                        group2ctx={"feat": mx.gpu(), "m_vec": mx.cpu()})
                    f_exec.forward()

                    forw_warp.append(f_exec.outputs[0])

                for i in range(len(forw_warp)):
                    weight = (1. * key_frame_interval - i) / key_frame_interval
                    forw_warp[i] = weight * forw_warp[i]

                # print 'forw_warp: ', len(forw_warp)

                back_warp = [feat_next]
                for i in range(key_frame_interval):
                    feat_sym = mx.sym.Variable(name="feat")
                    m_vec_sym = mx.sym.Variable(name="m_vec")

                    m_vec_grid = mx.sym.GridGenerator(data=m_vec_sym, transform_type='warp', name='m_vec_grid')
                    feat_warp = mx.sym.BilinearSampler(data=feat_sym, grid=m_vec_grid, name='warping_feat')

                    m_vec_data = data[idx + (key_frame_interval - i)][1]
                    b_exec = feat_warp.bind(ctx=mx.gpu(),
                        args={"feat": back_warp[-1], "m_vec": m_vec_data},
                        group2ctx={"feat": mx.gpu(), "m_vec": mx.cpu()})
                    b_exec.forward()

                    back_warp.append(b_exec.outputs[0])

                for i in range(len(back_warp)):
                    weight = (1. * key_frame_interval - i) / key_frame_interval
                    back_warp[i] = weight * back_warp[i]

                back_warp.reverse()
                # print 'back_warp: ', len(back_warp)

            else:
                print '\nframe {} (intermediate)'.format(idx)
                # print 'modulo {}'.format(idx % key_frame_interval)
                feat_forw = forw_warp[idx % key_frame_interval]
                feat_back = back_warp[idx % key_frame_interval]

                data_batch.data[0][-2] = feat_forw
                data_batch.provide_data[0][-2] = ('feat_forw', feat_forw.shape)
                data_batch.data[0][-1] = feat_back
                data_batch.provide_data[0][-1] = ('feat_back', feat_back.shape)
                # scores, boxes, data_dict, _ = im_detect(cur_predictor, data_batch, data_names, scales, config)
                output_all, _ = im_segment(cur_predictor, data_batch)
                output_all = [mx.ndarray.argmax(output['croped_score_output'], axis=1).asnumpy() for output in output_all]

            elapsed = toc()
            time += elapsed
            count += 1
            print 'testing {} {:.4f}s [{:.4f}s]'.format(im_name, elapsed, time/count)

            pred = np.uint8(np.squeeze(output_all))
            segmentation_result = Image.fromarray(pred)
            pallete = getpallete(256)
            segmentation_result.putpalette(pallete)
            _, im_filename = os.path.split(im_name)
            segmentation_result.save(output_dir + '/seg_' + im_filename)

            label = None
            if has_gt:
                # if annotation available for frame
                _, lb_filename = os.path.split(label_files[lb_idx])
                if im_filename[:len(ref_img_prefix)] == lb_filename[:len(ref_img_prefix)]:
                    print 'label {}'.format(lb_filename[:len(ref_img_prefix)])
                    label = np.asarray(Image.open(label_files[lb_idx]))
                    if lb_idx < len(label_files) - 1:
                        lb_idx += 1
            else:
                _, lb_filename = os.path.split(label_files[idx])
                print 'label {}'.format(lb_filename[:len(ref_pred_prefix)])
                label = np.asarray(Image.open(label_files[idx]))

            if label is not None:
                curr_hist = fast_hist(pred.flatten(), label.flatten(), num_classes)
                hist += curr_hist
                print 'mIoU {mIoU:.3f}'.format(
                    mIoU=round(np.nanmean(per_class_iu(curr_hist)) * 100, 2))
                print '(cum) mIoU {mIoU:.3f}'.format(
                    mIoU=round(np.nanmean(per_class_iu(hist)) * 100, 2))

    ious = per_class_iu(hist) * 100
    print ' '.join('{:.03f}'.format(i) for i in ious)
    print '===> final mIoU {mIoU:.3f}'.format(mIoU=round(np.nanmean(ious), 2))

    avg_diff_0 = diff_0 / (num_ex * interv)
    avg_diff_5 = diff_5 / (num_ex * interv)
    avg_diff_10 = diff_10 / (num_ex * interv)
    print avg_diff_0, avg_diff_5, avg_diff_10

    print 'done'

if __name__ == '__main__':
    main()
