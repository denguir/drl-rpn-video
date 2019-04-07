import tensorflow as tf
from nets.vgg16 import vgg16
from model.config import cfg, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
from model.train_val import train_net, get_training_roidb

# load model for detection
net = vgg16()

# load train set

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb



save_path = '/home/vador/Documents/project/AI/drl-rpn-tf-video/output-weights/voc_2007_train/'
# Cannot use weights_path because it leads to dimensions mismatch
weigths_path = '/home/vador/Documents/project/AI/drl-rpn-tf-video/pretrained-data/data3D/vgg16_drl_rpn_iter_1.ckpt'
imdb_name = 'voc_2007_train'
imdbval_name = 'voc_2007_trainval'

imdb, roidb = combined_roidb(imdb_name)
print('{:d} roidb entries'.format(len(roidb)))

output_dir = get_output_dir(imdb, None, save_path)

_, valroidb = combined_roidb(imdbval_name)

train_net(net, imdb, roidb, valroidb, output_dir, pretrained_model=weigths_path, max_iters=1)