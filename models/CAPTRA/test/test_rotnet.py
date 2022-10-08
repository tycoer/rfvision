from models.CAPTRA.models.captra_rotnet import CaptraRotNet
import torch
meta = {'path': ['/disk1/data/captar/render/train/2/bb7586ebee0dc2be4e346ee2650d150/0003/05.pkl',
              '/disk1/data/captar/render/train/2/ce905d4381d4daf65287b12a83c64b85/0002/64.pkl',
              '/disk1/data/captar/render/train/2/d2e1dc9ee02834c71621c7edb823fc53/0004/54.pkl',
              '/disk1/data/captar/render/train/2/d1addad5931dd337713f2e93cbeac35d/0005/04.pkl',
              '/disk1/data/captar/render/train/2/804092c488e7c8e420d3c05c08e26f/0000/87.pkl',
              '/disk1/data/captar/render/train/2/12ddb18397a816c8948bef6886fb4ac/0001/19.pkl',
              '/disk1/data/captar/render/train/2/3a7737c7bb4a6194f60bf3def77dca62/0005/98.pkl',
              '/disk1/data/captar/render/train/2/a95e0d8b37f8ca436a3309b77df3f951/0004/27.pkl',
              '/disk1/data/captar/render/train/2/d162f269bc5fb7da85df81f999944b5d/0005/62.pkl',
              '/disk1/data/captar/render/train/2/cda15ee9ad73f9d4661dc36b3dd991aa/0000/99.pkl',
              '/disk1/data/captar/render/train/2/3a7737c7bb4a6194f60bf3def77dca62/0002/56.pkl',
              '/disk1/data/captar/render/train/2/4967063fc3673caa47fe6b02985fbd0/0003/87.pkl'],
     'ori_path': ['/disk1/data/captar/nocs_full/train/13380/0000_composed.png',
                  '/disk1/data/captar/nocs_full/train/11709/0004_composed.png',
                  '/disk1/data/captar/nocs_full/train/21521/0003_composed.png',
                  '/disk1/data/captar/nocs_full/train/23472/0009_composed.png',
                  '/disk1/data/captar/nocs_full/train/04197/0004_composed.png',
                  '/disk1/data/captar/nocs_full/train/06065/0004_composed.png',
                  '/disk1/data/captar/nocs_full/train/25660/0007_composed.png',
                  '/disk1/data/captar/nocs_full/train/21344/0006_composed.png',
                  '/disk1/data/captar/nocs_full/train/25974/0003_composed.png',
                  '/disk1/data/captar/nocs_full/train/04331/0002_composed.png',
                  '/disk1/data/captar/nocs_full/train/10957/0006_composed.png',
                  '/disk1/data/captar/nocs_full/train/20098/0008_composed.png'],
     'nocs2camera': [{'rotation': torch.tensor([[[0.5450, -0.2357, -0.8046],
                                           [-0.4093, 0.7628, -0.5006],
                                           [0.7317, 0.6022, 0.3192]],

                                          [[-0.4053, 0.0040, -0.9142],
                                           [-0.6060, 0.7475, 0.2720],
                                           [0.6845, 0.6642, -0.3005]],

                                          [[-0.9759, -0.0748, 0.2051],
                                           [0.0319, 0.8807, 0.4727],
                                           [-0.2160, 0.4678, -0.8570]],

                                          [[-0.9983, 0.0014, -0.0577],
                                           [-0.0394, 0.7136, 0.6995],
                                           [0.0421, 0.7006, -0.7123]],

                                          [[-0.6778, -0.0446, -0.7339],
                                           [-0.4071, 0.8540, 0.3241],
                                           [0.6122, 0.5184, -0.5970]],

                                          [[-0.9755, -0.0626, 0.2111],
                                           [0.0700, 0.8208, 0.5669],
                                           [-0.2088, 0.5677, -0.7963]],

                                          [[-0.7942, 0.1286, 0.5939],
                                           [0.3411, 0.9032, 0.2606],
                                           [-0.5029, 0.4095, -0.7612]],

                                          [[0.9531, -0.0851, 0.2903],
                                           [0.1724, 0.9413, -0.2902],
                                           [-0.2486, 0.3267, 0.9119]],

                                          [[-0.7832, 0.0424, -0.6203],
                                           [-0.5036, 0.5418, 0.6729],
                                           [0.3646, 0.8394, -0.4030]],

                                          [[-0.9077, -0.3305, -0.2586],
                                           [-0.4144, 0.8034, 0.4277],
                                           [0.0664, 0.4953, -0.8662]],

                                          [[-0.9715, -0.0446, 0.2330],
                                           [0.0664, 0.8917, 0.4478],
                                           [-0.2277, 0.4505, -0.8633]],

                                          [[0.0060, 0.0112, -0.9999],
                                           [-0.3888, 0.9213, 0.0080],
                                           [0.9213, 0.3887, 0.0099]]], dtype=torch.float64),
                      'scale': torch.tensor([0.1872, 0.1927, 0.2278, 0.2352, 0.2287, 0.2216, 0.1935, 0.2265, 0.2290,
                                       0.2236, 0.2142, 0.2370], dtype=torch.float64),
                      'translation': torch.tensor([[[3.6360e-01],
                                              [1.7455e-02],
                                              [-1.2099e+00]],

                                             [[-6.2024e-01],
                                              [-2.0204e-01],
                                              [-1.2580e+00]],

                                             [[-3.6233e-01],
                                              [-5.7788e-02],
                                              [-1.5950e+00]],

                                             [[-2.0816e-01],
                                              [1.0217e-01],
                                              [-1.2000e+00]],

                                             [[-1.7790e-01],
                                              [2.6815e-02],
                                              [-1.0580e+00]],

                                             [[-1.5993e-01],
                                              [1.7228e-01],
                                              [-1.4530e+00]],

                                             [[-1.1606e+00],
                                              [4.2800e-01],
                                              [-2.1721e+00]],

                                             [[5.1945e-05],
                                              [-1.2419e-01],
                                              [-1.8550e+00]],

                                             [[-1.2892e-01],
                                              [3.1228e-01],
                                              [-1.0730e+00]],

                                             [[-2.2659e-01],
                                              [-3.4430e-01],
                                              [-1.1190e+00]],

                                             [[-2.0712e-01],
                                              [8.6382e-02],
                                              [-1.3691e+00]],

                                             [[6.0167e-01],
                                              [-3.5497e-01],
                                              [-1.2000e+00]]], dtype=torch.float64)}],
     'crop_pose': [{'translation': torch.tensor([[[0.3630],
                                            [0.0193],
                                            [-1.2124]],

                                           [[-0.6112],
                                            [-0.2052],
                                            [-1.2770]],

                                           [[-0.4052],
                                            [-0.0355],
                                            [-1.5559]],

                                           [[-0.1858],
                                            [0.0970],
                                            [-1.1888]],

                                           [[-0.1766],
                                            [-0.0026],
                                            [-1.0339]],

                                           [[-0.1556],
                                            [0.2085],
                                            [-1.4651]],

                                           [[-1.1504],
                                            [0.4555],
                                            [-2.1559]],

                                           [[0.0241],
                                            [-0.1198],
                                            [-1.8554]],

                                           [[-0.1288],
                                            [0.3068],
                                            [-1.0674]],

                                           [[-0.2294],
                                            [-0.3383],
                                            [-1.1205]],

                                           [[-0.2095],
                                            [0.0878],
                                            [-1.3694]],

                                           [[0.5868],
                                            [-0.3665],
                                            [-1.2046]]], dtype=torch.float64),
                    'scale': torch.tensor([0.2104, 0.1690, 0.2431, 0.2308, 0.2158, 0.2234, 0.2132, 0.2445, 0.1870,
                                     0.2302, 0.1905, 0.2459], dtype=torch.float64)}],
     'pre_fetched': {},
     'nocs_corners': torch.tensor([[[[-0.3245, -0.1987, -0.3244],
                               [0.3245, 0.1987, 0.3244]]],

                             [[[-0.3301, -0.1789, -0.3301],
                               [0.3301, 0.1789, 0.3301]]],

                             [[[-0.3481, -0.0878, -0.3481],
                               [0.3481, 0.0878, 0.3481]]],

                             [[[-0.3115, -0.2364, -0.3115],
                               [0.3115, 0.2364, 0.3115]]],

                             [[[-0.3415, -0.1294, -0.3415],
                               [0.3415, 0.1294, 0.3415]]],

                             [[[-0.3059, -0.1955, -0.3438],
                               [0.3059, 0.1955, 0.3438]]],

                             [[[-0.3335, -0.1660, -0.3335],
                               [0.3335, 0.1660, 0.3335]]],

                             [[[-0.3345, -0.1612, -0.3349],
                               [0.3345, 0.1612, 0.3349]]],

                             [[[-0.3207, -0.2106, -0.3207],
                               [0.3207, 0.2106, 0.3207]]],

                             [[[-0.3457, -0.1051, -0.3457],
                               [0.3457, 0.1051, 0.3457]]],

                             [[[-0.3335, -0.1660, -0.3335],
                               [0.3335, 0.1660, 0.3335]]],

                             [[[-0.3409, -0.1326, -0.3409],
                               [0.3409, 0.1326, 0.3409]]]], dtype=torch.float64),
     'points_mean': torch.tensor([[[0.3619],
                             [-0.0050],
                             [-1.1981]],

                            [[-0.6007],
                             [-0.2010],
                             [-1.2727]],

                            [[-0.3994],
                             [-0.0892],
                             [-1.5806]],

                            [[-0.1921],
                             [0.0938],
                             [-1.2070]],

                            [[-0.1757],
                             [-0.0047],
                             [-1.0333]],

                            [[-0.1941],
                             [0.2128],
                             [-1.4436]],

                            [[-1.1891],
                             [0.4804],
                             [-2.2033]],

                            [[0.0153],
                             [-0.1132],
                             [-1.8705]],

                            [[-0.1262],
                             [0.2964],
                             [-1.0486]],

                            [[-0.2105],
                             [-0.3508],
                             [-1.1159]],

                            [[-0.2099],
                             [0.0784],
                             [-1.3639]],

                            [[0.5543],
                             [-0.3655],
                             [-1.2068]]], dtype=torch.float64)}


obj_cfg =   {'1': {'name': 'bottle',
   'sym': True,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '2a9817a43c5b3983bb13793251b29587',
   'bad_ins': []},
  '2': {'name': 'bowl',
   'sym': True,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '1b4d7803a3298f8477bdcb8816a3fac9',
   'bad_ins': []},
  '3': {'name': 'camera',
   'sym': False,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '5d42d432ec71bfa1d5004b533b242ce6',
   'bad_ins': ['1298634053ad50d36d07c55cf995503e',
    '2153bc743019671ae60635d9e388f801',
    '22217d5660444eeeca93934e5f39869',
    '290abe056b205c08240c46d333a693f',
    '39419e462f08dcbdc98cccf0d0f53d7',
    '4700873107186c6e2203f435e9e6785',
    '550aea46c75351a387cfe978d99ba05d',
    '60923e8a6c785a8755a834a7aafb0236',
    '6ed69b00b4632b6e07718ee10b83e10',
    '7077395b60bf4aeb3cb44973ec1ffcf8',
    '87b8cec4d55b5f2d75d556067d060edf',
    '97cd28c085e3754f22c69c86438afd28',
    'a9408583f2c4d6acad8a06dbee1d115',
    'b27815a2bde54ad3ab3dfa44f5fab01',
    'b42c73b391e14cb16f05a1f780f1cef',
    'c3e6564fe7c8157ecedd967f62b864ab',
    'c802792f388650428341191174307890',
    'd680d61f934eaa163b211460f022e3d9',
    'd9bb9c5a48c3afbfb84553e864d84802',
    'e3dc17dbde3087491a722bdf095986a4',
    'e57aa404a000df88d5d4532c6bb4bd2b',
    'eb86c8c2a20066d0fb1468f5fc754e02',
    'ee58b922bd93d01be4f112f1b3124b84',
    'fe669947912103aede650492e45fb14f',
    'ff74c4d2e710df3401a67448bf8fe08']},
  '4': {'name': 'can',
   'sym': True,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '97ca02ee1e7b8efb6193d9e76bb15876',
   'bad_ins': []},
  '5': {'name': 'laptop',
   'sym': False,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': '3e9af28eb2d6e216a4e3429ccb8eaf16',
   'bad_ins': []},
  '6': {'name': 'mug',
   'sym': False,
   'type': 'revolute',
   'num_parts': 1,
   'num_joints': 0,
   'tree': [-1],
   'parts_map': [[0]],
   'augment_idx': {'x': [-1], 'y': [-1], 'z': [-1]},
   'main_axis': [],
   'template': 'a6d9f9ae39728831808951ff5fb582ac',
   'bad_ins': []}}


if __name__ == '__main__':
    m = CaptraRotNet(obj_cfg=obj_cfg['1'],
                     backbone_out_dim=128,).cuda()
    data = dict(points=torch.rand(12, 3, 4096).cuda(),
                labels=torch.ones(12, 4096).cuda(),
                nocs=torch.rand(12, 3, 4096).cuda(),
                meta=meta)
    res = m(return_loss = True, **data)

