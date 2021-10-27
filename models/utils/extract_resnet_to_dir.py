import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import tqdm
import torch
import os
from PIL import Image
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='data/2.1.0')
    parser.add_argument('--img_folder', help='folder containing raw images', default='raw_images')
    parser.add_argument('--out_folder', help='output data folder', default='data/json_feat_2.1.0')

    parser.add_argument('--batch', help='batch size', default=256, type=int)
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--skip_existing', help='skip folders that already have feats', action='store_true')
    parser.add_argument('--visual_model', default='resnet18', help='model type', choices=['maskrcnn', 'resnet18', 'resnet50', 'resnet50_clip'])

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

    # load resnet model
    extractor = Resnet(args, eval=True)
    skipped = []

    for root, dirs, files in tqdm.tqdm(os.walk(args.data)):
        if os.path.basename(root) == args.img_folder:
            fimages = sorted([os.path.join(root, f) for f in files
                              if (f.endswith('.png') or (f.endswith('.jpg')))])
            if len(fimages) > 0:
                output_file = os.path.join(
                    root.replace(args.data, args.out_folder).replace('raw_images', ''),
                    filename
                )
                if args.skip_existing and os.path.isfile(output_file):
                    continue
                try:
                    print('{}'.format(root))
                    image_loader = Image.open if isinstance(fimages[0], str) else Image.fromarray
                    images = [image_loader(f) for f in fimages]
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
