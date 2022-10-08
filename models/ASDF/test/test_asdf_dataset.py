from models.ASDF.datasets.asdf_dataset import SDFSamples




if __name__ == '__main__':
    specs = {
        # model para
        "CodeLength": 253,
        "Articulation": True,
        "NumAtcParts": 1,
        "TrainWithParts": False,
        "ClampingDistance": 0.1,
        # dataset para
        'TestSplit': 'models/ASDF/examples/splits/sm_door_6_angle_test.json',
        'TrainSplit': 'models/ASDF/examples/splits/sm_door_6_angle_train.json',
        # meta
        'Class': 'door',
        "NumInstances": 92,
    }

    data_cfg = dict(
        data_source='./data',
        split='models/ASDF/examples/splits/sm_door_6_angle_train.json',
        subsample=16000,
        load_ram=False,
        articulation=specs["Articulation"],
        num_atc_parts=specs["NumAtcParts"],
    )

    dataset = SDFSamples(**data_cfg)
    data = dataset[0]