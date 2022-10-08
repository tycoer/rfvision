import argparse
import torch
import os

def converter(offical_state_dict):
    state_dict = {}
    for k, v in offical_state_dict.items():
        k = k.split('.')
        k.pop(1)
        k = '.'.join(k)
        state_dict[k] = v
    return state_dict

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Convert offical checkpoint file to rfvision-style.')
    arg_parser.add_argument('--checkpoint', required=True, type=str)
    args = arg_parser.parse_args()

    rf_checkpoint_path = args.checkpoint
    checkpoint = torch.load(rf_checkpoint_path)
    state_dict = checkpoint['state_dict']
    epoch = checkpoint['meta']['epoch']
    state_dict = converter(checkpoint['model_state_dict'])
    checkpoint_rf = dict(meta={'epoch': epoch},
                         optimizer={},
                         state_dict=state_dict)
    save_path = os.path.join(os.path.dirname(rf_checkpoint_path), f'rf_{os.path.basename(rf_checkpoint_path)}')
    torch.save(checkpoint_rf, save_path)
    print(f'rfvision-style checkpoint is saved to {save_path}')
