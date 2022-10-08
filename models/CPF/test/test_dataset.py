from models.CPF.datasets.hodata import HOdata

if __name__ == '__main__':
    dataset_all = HOdata(data_root='/hdd2/data/handata',
                         assets_root='/home/hanyang/rfvision/models/CPF/assets',
                         njoints=21
                         )
    dataset = dataset_all.get_dataset(dataset='ho3d',
                                      data_root='/hdd2/data/handata',
                                      data_split='train',
                                      split_mode='objects',
                                      use_cache=False,
                                      mini_factor=1,
                                      center_idx=9,
                                      enable_contact=False,
                                      filter_no_contact=False,
                                      filter_thresh=0.,
                                      synt_factor=(0, 5),
                                      assets_root='/home/hanyang/rfvision/models/CPF/assets'
                                      )


    data = dataset[0]