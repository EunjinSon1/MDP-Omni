# utils.dbhelper.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
# Modified by: Eunjin Son (eunjinson@jbnu.ac.kr)
# - Added support for OmniPus.
#
import os
import os.path as osp

import yaml

from utils.common import *
from utils.image import *
from utils.log import *
from utils.ocam import OcamModel as SyntheticOcamModel
from utils.ocam_omnipus import OcamModel as OmnipusOcamModel


def is_omnipus_dbname(dbname: str) -> bool:
    return dbname.lower().startswith("omnipus")


def loadDBConfigs(dbname: str, dbpath: str, opts: Edict):
    if is_omnipus_dbname(dbname):
        ocams = _load_omnipus_ocams(dbpath)
    else:
        opts, ocams = _load_synthetic_configs(dbpath, opts)

    func = "__load_train_%s(opts)" % dbname
    try:
        opts = eval(func)
        LOG_INFO('Found "%s" training configs' % dbname)
        opts.dtype = "gt"
    except Exception as exc:
        LOG_INFO('Training configs not found "%s" (%s)' % (dbname, exc))
    finally:
        return opts, ocams


def _load_synthetic_configs(dbpath: str, opts: Edict):
    config_file = osp.join(dbpath, "config.yaml")
    config = yaml.safe_load(open(config_file))

    for key in config["config"].keys():
        opts[key] = config["config"][key]
    opts.min_depth = opts.omnimvs_sweep_min_depth
    for key in config["dataset"].keys():
        opts[key] = config["dataset"][key]

    cameras_cfg = config["cameras"]
    ocams = []
    for i in range(4):
        ocam = SyntheticOcamModel()
        ocam.setConfig(cameras_cfg[i])
        mask_file = osp.join(dbpath, ocam.invalid_mask_file)
        if not osp.exists(mask_file):
            ocam.invalid_mask = ocam.makeInvisibleMask()
            writeImage(ocam.invalid_mask, mask_file)
        ocam.invalid_mask = readImage(mask_file).astype(np.bool_)
        ocams.append(ocam)
    return opts, ocams


def _load_omnipus_ocams(dbpath: str):
    shared_calib_dir = osp.join(osp.dirname(dbpath), "calibration")
    ocams = []
    for camera_idx in range(1, 5):
        cam_file = osp.join(shared_calib_dir, f"cam{camera_idx}.yaml")
        cam_cfg = yaml.safe_load(open(cam_file))
        ocam = OmnipusOcamModel()
        ocam.setConfig(cam_cfg)
        mask_file = osp.join(shared_calib_dir, f"cam{camera_idx}_mask.png")
        if not osp.exists(mask_file):
            ocam.invalid_mask = ocam.makeInvisibleMask()
            writeImage(ocam.invalid_mask, mask_file)
        ocam.invalid_mask = readImage(mask_file).astype(np.bool_)
        ocams.append(ocam)
    return ocams



def collect_omnipus_sequence_frames(base_path, seq_names):
    sequence_frames = {}

    for seq_name in seq_names:
        seq_dir = os.path.join(base_path, seq_name, "omnidepth_gt")
        tiff_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(".tiff")])
        frame_ids = [int(os.path.splitext(f)[0]) for f in tiff_files]
        sequence_frames[seq_name] = frame_ids
        print(f"{seq_name}: total={len(frame_ids)}")

    return sequence_frames


def __load_train_omnipusa(opts):
    base_dir = "../../omnidata/omnipusa"

    opts.train_sequences = collect_omnipus_sequence_frames(
        base_dir,
        [
            "omnipusa_case2",
            "omnipusa_case3",
            "omnipusa_case4",
            "omnipusa_case5",
            "omnipusa_case6",
            "omnipusa_case7",
            "omnipusa_case11",
            "omnipusa_case12",
            "omnipusa_case13",
            "omnipusa_case15",
        ],
    )
    opts.test_sequences = collect_omnipus_sequence_frames(
        base_dir,
        [
            "omnipusa_case1",
            "omnipusa_case8",
            "omnipusa_case9",
            "omnipusa_case10",
            "omnipusa_case14",
        ],
    )
    _print_frame_count_summary(opts)
    return opts


def __load_train_omnipusb(opts):
    base_dir = "../../omnidata/omnipusb"

    opts.train_sequences = collect_omnipus_sequence_frames(
        base_dir,
        [
            "omnipusb_case1",
            "omnipusb_case2",
            "omnipusb_case3",
            "omnipusb_case4",
            "omnipusb_case5",
            "omnipusb_case7",
            "omnipusb_case10",
            "omnipusb_case12",
            "omnipusb_case13",
        ],
    )
    opts.test_sequences = collect_omnipus_sequence_frames(
        base_dir,
        [
            "omnipusb_case6",
            "omnipusb_case8",
            "omnipusb_case9",
            "omnipusb_case11",
        ],
    )
    _print_frame_count_summary(opts)
    return opts


def __load_train_omnipusc(opts):
    base_dir = "../../omnidata/omnipusc"

    opts.train_sequences = collect_omnipus_sequence_frames(
        base_dir,
        [
            "omnipusc_case2",
            "omnipusc_case3",
            "omnipusc_case5",
        ],
    )
    opts.test_sequences = collect_omnipus_sequence_frames(
        base_dir,
        [
            "omnipusc_case1",
            "omnipusc_case4",
        ],
    )
    _print_frame_count_summary(opts)
    return opts


def _print_frame_count_summary(opts):
    total_train = sum(len(v) for v in opts.train_sequences.values())
    total_test = sum(len(v) for v in opts.test_sequences.values())
    print("\n====== Frame Count Summary ======")
    print(f"Train frames : {total_train}")
    print(f"Test frames  : {total_test}")
    print(f"Total        : {total_train + total_test}")
    print("=================================\n")


def __load_train_sunny(opts):
    opts.train_idx = list(range(1, 701))
    opts.test_idx = list(range(701, 1001))
    opts.gt_phi = 45
    return opts


__load_train_sunset = __load_train_cloudy = __load_train_sunny


def __load_train_omnithings(opts):
    opts.train_idx = list(range(1, 4097)) + list(range(5121, 10241))
    opts.test_idx = list(range(4097, 5121))
    opts.gt_phi = 90
    return opts


def __load_train_omnihouse(opts):
    opts.train_idx = list(range(1, 2049))
    opts.test_idx = list(range(2049, 2561))
    opts.gt_phi = 90
    return opts
