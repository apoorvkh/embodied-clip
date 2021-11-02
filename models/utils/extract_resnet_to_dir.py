import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

from tqdm import tqdm
import torch
import os
import json
from PIL import Image
import numpy as np
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='data/2.1.0')
    parser.add_argument('--img_folder', help='folder containing raw images', default='raw_images')
    parser.add_argument('--out_folder', help='output data folder', default='data/json_feat_2.1.0')
    parser.add_argument('--all_images', help='extract features for all images', action='store_true')

    parser.add_argument('--batch', help='batch size', default=256, type=int)
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--skip_existing', help='skip folders that already have feats', action='store_true')
    parser.add_argument('--visual_model', default='resnet18', help='model type', choices=['maskrcnn', 'resnet18', 'resnet50', 'resnet50_clip'])
    parser.add_argument('--augmentation', default=0, help='augmentation (1: color swap, 2: autoaugment)', type=int)

    # parser
    args = parser.parse_args()

    if args.visual_model == 'resnet18':
        filename = 'feat_conv.pt'
    elif args.visual_model == 'maskrcnn':
        filename = 'feat_conv_maskrcnn.pt'
    elif args.visual_model == 'resnet50':
        filename = 'feat_conv_resnet50.pt'
    elif args.visual_model == 'resnet50_clip':
        filename = 'feat_conv_resnet50_clip.pt'

    if args.augmentation:
        filename = filename.replace('feat_conv', 'feat_conv_colorSwap{}'.format(args.augmentation))

    # load resnet model
    extractor = Resnet(args, eval=True, autoaugment=(args.augmentation == 2))
    skipped = []

    for root, dirs, files in tqdm(os.walk(args.data)):
        if os.path.basename(root) == args.img_folder:

            if args.all_images is False:
                with open(root.replace(args.img_folder, 'traj_data.json'), 'r', encoding='utf-8') as file:
                    traj_data = json.load(file)
                files = []
                seen_low_indices = set([])
                for image_i, image_data in enumerate(traj_data['images']):
                    # Get first frame per low-level action and stop frame
                    if image_data['low_idx'] not in seen_low_indices or image_i == len(traj_data['images']) - 1:
                        seen_low_indices.add(image_data['low_idx'])
                        image_name = image_data['image_name'].replace('.png', '.jpg')
                        files.append(image_name)

            fimages = sorted([os.path.join(root, f) for f in files
                              if (f.endswith('.png') or (f.endswith('.jpg')))])

            if len(fimages) > 0:
                output_file = os.path.join(
                    root.replace(args.data, args.out_folder).replace(args.img_folder, ''),
                    filename
                )
                if args.skip_existing and os.path.isfile(output_file):
                    continue
                try:
                    print('{}'.format(root))
                    image_loader = Image.open if isinstance(fimages[0], str) else Image.fromarray
                    images = [image_loader(f) for f in fimages]
                    if args.augmentation == 1:
                        images = [Image.fromarray(np.asarray(image)[:,:,np.random.permutation(3)]) for image in images]
                    feat = extractor.featurize(images, batch=args.batch)
                    torch.save(feat.cpu(), output_file)
                except Exception as e:
                    print(e)
                    print("Skipping " + root)
                    skipped.append(root)

            else:
                print('empty; skipping {}'.format(root))

    print("Skipped:")
    print(skipped)
