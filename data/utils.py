# Copyright (c) Ruopeng Gao. All Rights Reserved.
from utils.nested_tensor import NestedTensor, tensor_list_to_nested_tensor

def collate_fn(batch):
    collated_batch = {
        "images": [],
        "infos": []
    }
    for data in batch:
        # collated_batch["images"].append(data["images"])
        collated_batch["images"].append(data["images"])
        # collated_batch["infos"].append(data["infos"])
        collated_batch["infos"].append(data["infos"])
    collated_batch["nested_tensors"] = tensor_list_to_nested_tensor([_ for seq in collated_batch["images"] for _ in seq])
    shape = collated_batch["nested_tensors"].tensors.shape
    b = len(batch)
    t = len(collated_batch["images"][0])
    collated_batch["nested_tensors"].tensors = collated_batch["nested_tensors"].tensors.reshape(
        b, t, shape[1], shape[2], shape[3]
    )
    collated_batch["nested_tensors"].mask = collated_batch["nested_tensors"].mask.reshape(
        b, t, shape[2], shape[3]
    )
    # collated_batch[]
    return collated_batch

def collate_fn_ref_kitti(batch):
    collated_batch = {
        "images": [],
        "infos": [],
        "sentences": [],
        "dataset_name": []
    }
    for i, data in enumerate(batch):
        collated_batch["images"].append(data["imgs"])
        # collated_batch["infos"].append({
        #     "boxes": data["gt_instances"][i].boxes,
        #     "labels": data["gt_instances"][i].labels,
        #     "ids": data["gt_instances"][i].obj_ids,
        #     "is_ref": data["gt_instances"][i].is_ref,
        #     "areas": data["gt_instances"][i].area
        # })
        collated_batch["infos"].append(data["infos"])
        collated_batch["sentences"].append(data["sentences"][0] if len(data["sentences"]) == 1 else data["sentences"])
        collated_batch["dataset_name"].append(data["dataset_name"])
    
    collated_batch["nested_tensors"] = tensor_list_to_nested_tensor([_ for seq in collated_batch["images"] for _ in seq]) 
    shape = collated_batch["nested_tensors"].tensors.shape

    b, t = len(batch), len(collated_batch["images"][0])

    collated_batch["nested_tensors"].tensors = collated_batch["nested_tensors"].tensors.reshape(
        b, t, shape[1], shape[2], shape[3]
    )
    collated_batch["nested_tensors"].mask = collated_batch["nested_tensors"].mask.reshape(
        b, t, shape[2], shape[3]
    )

    return collated_batch
