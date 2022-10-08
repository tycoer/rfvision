import torch
import os
import argparse



def count_weights(state_dict):
    total_para = 0
    for i in state_dict.values():
        total_para += i.numel()
    return total_para

def main(offical_ckpt_path, out_path, rf_keys, network_type='rot'):
    offical_ckpt = torch.load(offical_ckpt_path)

    weights = list(offical_ckpt['model'].values())
    keys = list(offical_ckpt['model'].keys())

    conv_interval = 2
    bn_interval = 5
    conv_start = 0
    bn_start = 18
    fp_length = 13

    rf_weights = []
    while True:
        # sa1
        rf_weights.extend(weights[conv_start: conv_start + conv_interval])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if bn_start >= 62:
            conv_start = 63
            bn_start = 75
            break

    while True:
        # sa2
        rf_weights.extend(weights[conv_start: conv_start + conv_interval])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval

        if bn_start >= 104:
            conv_start = 105
            bn_start = 111
            break

    while True:
        # sa3
        rf_weights.extend(weights[conv_start: conv_start + conv_interval])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval

        if bn_start >= 125:
            break

    if network_type == 'rot': # rotnet
        conv_start = 126
        bn_start = 130
    else: # coordnet
        num_gen = 0
        for k in keys:
            if 'gen' in k:
                num_gen += 1
        fp_start = bn_start
        conv_start = bn_start + num_gen
        bn_start = conv_start + 4


    while True:
        # fp3
        weights[conv_start] = weights[conv_start].unsqueeze(-1)
        rf_weights.extend([weights[conv_start]])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if network_type == 'rot':
            if bn_start >= 139:
                conv_start = 140
                bn_start = 144
                break
        else:
            if bn_start >= fp_start + num_gen + fp_length:
                conv_start = fp_start + num_gen + fp_length + 1
                bn_start = conv_start + 4
                break

    while True:
        # fp2
        weights[conv_start] = weights[conv_start].unsqueeze(-1)
        rf_weights.extend([weights[conv_start]])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if network_type == 'rot':
            if bn_start >= 153:
                conv_start = 154
                bn_start = 158
                break
        else:
            if bn_start >= fp_start + num_gen + fp_length * 2 + 1:
                conv_start = fp_start + num_gen + (fp_length + 1) * 2
                bn_start = conv_start + 4
                break


            # if bn_start >= 201:
            #     conv_start = 202
            #     bn_start = 206
            #     break

    while True:
        # fp1
        weights[conv_start] = weights[conv_start].unsqueeze(-1)
        rf_weights.extend([weights[conv_start]])
        conv_start = conv_start + conv_interval

        rf_weights.extend(weights[bn_start: bn_start + bn_interval])
        bn_start = bn_start + bn_interval
        if network_type == 'rot':
            if bn_start >= 167:
                break
        else:
            if bn_start >= fp_start + num_gen + fp_length * 3 + 2:
                bn_start = fp_start + num_gen + fp_length * 3 + 3
                break

    if network_type == 'rot': # rotnet
        for i in range(168, len(weights)):
            rf_weights.append(weights[i])
    else: # coordnet
        for i in range(bn_start, len(weights)):
            rf_weights.append(weights[i])

    assert len(rf_keys) ==  len(rf_weights)

    rf_ckpt = dict(
        state_dict = {k: v for k, v in zip(rf_keys, rf_weights)},
        meta={'epoch': 250},
        optimizer={})
    torch.save(rf_ckpt, out_path)
    print(f'rfvision-style checkpoint is saved to {out_path}')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Convert offical checkpoint file to rfvision-style.')
    arg_parser.add_argument('--checkpoint', required=True, type=str, help='offical checkpoint file path.',
                            default='/home/hanyang/CAPTRA-main/6_mug_rot/ckpt/model_0000.pt')
    arg_parser.add_argument('--out_path', type=str, help='out path of rfvision-style checkpoint.', required=True)
    args = arg_parser.parse_args()

    main(args.checkpoint, args.out_path)