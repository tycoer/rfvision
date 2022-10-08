import sys
import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from termcolor import colored, cprint
import json
from ..datasets.cionline import CIOnline
from ..datasets.cidata import CIAdaptQueries, CIDumpedQueries

from .ocl_loader import OptimzedContentLoader
from .dep import process_sample

from tqdm import tqdm


def main(
    cidata_path,
    optimized_content_path,
    save_prefix,
    hodata_path,
    anchor_path,
    hodata_use_cache,
    hodata_center_idx,
    wait_time,
    sample_vis_freq,
    use_gui,
    n_workers,
    vhacd_exe,
):
    ci_online = CIOnline(
        cidata_path, hodata_path, anchor_path, hodata_use_cache=hodata_use_cache, hodata_center_idx=hodata_center_idx
    )
    ocl = OptimzedContentLoader(optimized_content_path)
    assert len(ci_online) == len(ocl)
    cprint(f"Got all samples, with len {len(ci_online)} and {len(ocl)} !", "cyan")

    os.makedirs(save_prefix, exist_ok=True)
    save_gif_folder_dict = {}
    save_obj_folder_dict = {}
    for stage in ["gt", "honet", "cpf"]:
        os.makedirs(os.path.join(save_prefix, stage), exist_ok=True)
        save_gif_folder = os.path.join(save_prefix, stage, "save_gifs")
        save_obj_folder = os.path.join(save_prefix, stage, "save_objs")
        os.makedirs(save_gif_folder, exist_ok=True)
        os.makedirs(save_obj_folder, exist_ok=True)
        save_gif_folder_dict[stage] = save_gif_folder
        save_obj_folder_dict[stage] = save_obj_folder

    # get task list
    task_list_dict = {
        "gt": [],
        "honet": [],
        "cpf": [],
    }
    counter = 0
    for sample_idx in tqdm(range(len(ocl))):
        ci_item = ci_online[sample_idx]
        ocl_item = ocl[sample_idx]

        if "pour_milk" in ci_item[CIAdaptQueries.IMAGE_PATH]:
            continue

        counter += 1

        # ============ gt task >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        gt_sample_info = {
            "hand_verts": ci_online[sample_idx][CIAdaptQueries.HAND_VERTS_3D],
            "obj_verts": ci_online[sample_idx][CIAdaptQueries.OBJ_VERTS_3D],
            "hand_faces": ci_online[sample_idx][CIAdaptQueries.HAND_FACES],
            "obj_faces": ci_online[sample_idx][CIAdaptQueries.OBJ_FACES],
        }
        task_list_dict["gt"].append(
            delayed(process_sample)(
                sample_idx,
                gt_sample_info,
                save_gif_folder=save_gif_folder_dict["gt"],
                save_obj_folder=save_obj_folder_dict["gt"],
                use_gui=use_gui,
                wait_time=wait_time,
                sample_vis_freq=sample_vis_freq,
                vhacd_exe=vhacd_exe,
            )
        )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ honet task >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        honet_sample_info = {
            "hand_verts": ci_online[sample_idx][CIDumpedQueries.HAND_VERTS_3D],
            "obj_verts": ci_online[sample_idx][CIDumpedQueries.OBJ_VERTS_3D],
            "hand_faces": ci_online[sample_idx][CIAdaptQueries.HAND_FACES],
            "obj_faces": ci_online[sample_idx][CIAdaptQueries.OBJ_FACES],
        }
        task_list_dict["honet"].append(
            delayed(process_sample)(
                sample_idx,
                honet_sample_info,
                save_gif_folder=save_gif_folder_dict["honet"],
                save_obj_folder=save_obj_folder_dict["honet"],
                use_gui=use_gui,
                wait_time=wait_time,
                sample_vis_freq=sample_vis_freq,
                vhacd_exe=vhacd_exe,
            )
        )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ============ cpf task >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        cpf_sample_info = ocl_item.copy()
        cpf_sample_info.update(
            {
                "hand_faces": ci_online[sample_idx][CIAdaptQueries.HAND_FACES],
                "obj_faces": ci_online[sample_idx][CIAdaptQueries.OBJ_FACES],
            }
        )
        task_list_dict["cpf"].append(
            delayed(process_sample)(
                sample_idx,
                cpf_sample_info,
                save_gif_folder=save_gif_folder_dict["cpf"],
                save_obj_folder=save_obj_folder_dict["cpf"],
                use_gui=use_gui,
                wait_time=wait_time,
                sample_vis_freq=sample_vis_freq,
                vhacd_exe=vhacd_exe,
            )
        )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    res_dict_raw = {}
    for stage in ["gt", "honet", "cpf"]:
        res_dict_raw[stage] = Parallel(n_jobs=n_workers, verbose=1)(task_list_dict[stage])

    # get results
    cprint(f"simulate {counter} samples!")
    res_dict = {}
    for stage in ["gt", "honet", "cpf"]:
        distances = res_dict_raw[stage]
        res_dict[stage] = {
            "mean_dist": np.mean(distances),
            "std": np.std(distances),
        }

    # save computed results
    with open(os.path.join(save_prefix, "res.json"), "w") as outfile:
        json.dump(
            res_dict,
            outfile,
            indent=4,
        )
    with open(os.path.join(save_prefix, "res_raw.json"), "w") as outfile:
        json.dump(
            res_dict_raw,
            outfile,
        )


if __name__ == "__main__":
    # ============ argument parsing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cidata_path", type=str, required=True)
    parser.add_argument("--optimized_content_path", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, required=True)

    parser.add_argument("--hodata_path", type=str, default="data")
    parser.add_argument("--anchor_path", type=str, default="assets/anchor")
    parser.add_argument("--hodata_use_cache", action="store_true")
    parser.add_argument("--hodata_no_use_cache", action="store_true")
    parser.add_argument("--hodata_center_idx", type=int, default=9)

    parser.add_argument("--wait_time", type=float, default=0.0)
    parser.add_argument("--sample_vis_freq", type=int, default=10)
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--n_workers", type=int, default=10)
    parser.add_argument("--vhacd_exe", type=str, default="/home/xinyu/v-hacd/build/linux/test/testVHACD")

    args = parser.parse_args()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    cidata_path = args.cidata_path
    optimized_content_path = args.optimized_content_path
    save_prefix = args.save_prefix

    hodata_path = args.hodata_path
    anchor_path = args.anchor_path

    # deal with dual flags
    if not args.hodata_use_cache and not args.hodata_no_use_cache:
        g_hodata_use_cache = True
    elif not args.hodata_use_cache and args.hodata_no_use_cache:
        g_hodata_use_cache = False
    elif args.hodata_use_cache and not args.hodata_no_use_cache:
        g_hodata_use_cache = True
    else:
        g_hodata_use_cache = True

    hodata_center_idx = args.hodata_center_idx

    wait_time = args.wait_time
    sample_vis_freq = args.sample_vis_freq
    use_gui = args.use_gui
    n_workers = args.n_workers
    vhacd_exe = args.vhacd_exe

    main(
        cidata_path=cidata_path,
        optimized_content_path=optimized_content_path,
        save_prefix=save_prefix,
        hodata_path=hodata_path,
        anchor_path=anchor_path,
        hodata_use_cache=g_hodata_use_cache,
        hodata_center_idx=hodata_center_idx,
        wait_time=wait_time,
        sample_vis_freq=sample_vis_freq,
        use_gui=use_gui,
        n_workers=n_workers,
        vhacd_exe=vhacd_exe,
    )
