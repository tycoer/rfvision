from models.Garmentnets.datasets.garmentnet_dataset import ConvImplicitWNFDataset

if __name__ == '__main__':
    dataset = ConvImplicitWNFDataset(
        zarr_path='data/garmentnets_dataset_sample.zarr/Tshirt',  # data root
        metadata_cache_dir='~/local/.cache/metadata_cache_dir',
        # sample size
        batch_size=24,
        num_workers=0,
        # sample size
        num_pc_sample=6000,
        num_volume_sample=6000,
        num_surface_sample=6000,
        num_mc_surface_sample=0,
        # mixed sampling config
        surface_sample_ratio=0,
        surface_sample_std=0.05,
        # surface sample noise
        # use 0.5
        surface_normal_noise_ratio=0,
        surface_normal_std=0.01,
        # data augumentation
        enable_augumentation=True,
        random_rot_range=[-180, 180],
        num_views=4,
        # volume
        volume_size=128,
        # or nocs_signed_distance_field or nocs_occupancy_grid or sim_nocs_winding_number_field or nocs_distance_field
        volume_group='nocs_winding_number_field',
        # use 0.05
        tsdf_clip_value=None,
        volume_absolute_value=False,
        include_volume=False,
        # random seed
        static_epoch_seed=False,
        # datamodule config
        dataset_split=[8, 1, 1],
        split_seed=0
    )
    data = dataset[0]