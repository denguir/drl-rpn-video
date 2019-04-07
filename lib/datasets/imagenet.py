# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from datasets.imdb import imdb
import os, sys
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess

class imagenet(imdb):
    def __init__(self, image_set, devkit_path):
        super(imagenet, self).__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'ILSVRC2013_DET_' + self._image_set[:-1])
        synsets = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_det.mat'))
        self._classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
        self._wnid = (0,)
        for i in range(200):
            self._classes = self._classes + (synsets['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets['synsets'][0][i][1][0],)
        self._wnid_to_ind = dict(zip(self._wnid, range(self.num_classes)))
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = ['.JPEG']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                              index[:23] + self._image_ext[0])
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._devkit_path, 'data', 'det_lists', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._devkit_path, 'ILSVRC2013_DET_bbox_' + self._image_set[:-1], index[:23] + '.xml')
        # print 'Loading: {}'.format(filename) 
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            #x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            #y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            #x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            #y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_imagenet_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', comp_id + '_')
        print('Writing {} results file'.format(self.name))
        filename = path + 'det_' + self._image_set + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(im_ind + 1, cls_ind, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    # def _do_matlab_eval(self, comp_id, output_dir='output'):
    #     rm_results = self.config['cleanup']

    #     path = os.path.join(os.path.dirname(__file__),
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'setenv(\'LC_ALL\',\'C\'); imagenet_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
    #            .format(self._devkit_path, comp_id,
    #                    self._image_set, output_dir, int(rm_results))
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    # def evaluate_detections(self, all_boxes, output_dir):
    #     comp_id = self._write_imagenet_results_file(all_boxes)
    #     self._do_matlab_eval(comp_id, output_dir)

    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True

if __name__ == '__main__':
    d = imagenet('val1', '')
    res = d.roidb
    from IPython import embed; embed()