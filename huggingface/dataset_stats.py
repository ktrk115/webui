import json
from pathlib import Path

from huggingface_hub import snapshot_download


def get_page_ids(name):
    cache_dir = Path(f"./cache/{name}")
    snapshot_download(
        repo_id=f"biglab/{name}",
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns="*.json",
    )

    # load json file
    json_path = list(cache_dir.glob("*.json"))[0]
    with json_path.open() as f:
        page_ids = json.load(f)

    return page_ids


train_splits = [
    "webui-7k",
    "webui-7kbal",
    "webui-70k",
    "webui-350k",
]

val_split = "webui-val"
test_split = "webui-test"

val_ids = get_page_ids(val_split)
test_ids = get_page_ids(test_split)


for name in train_splits + [val_split, test_split]:
    page_ids = get_page_ids(name)
    duplicates = len(page_ids) - len(set(page_ids))
    val_conflict_ids = set(page_ids) & set(val_ids)
    test_conflict_ids = set(page_ids) & set(test_ids)

    print(f"{name}: {len(page_ids)} pages, {duplicates} duplicates")
    if not name.endswith("val"):
        sample_ids = sorted(list(val_conflict_ids))[:3]
        print(f"\t{len(val_conflict_ids)} val conflicts: {sample_ids}")
    if not name.endswith("test"):
        sample_ids = sorted(list(test_conflict_ids))[:3]
        print(f"\t{len(test_conflict_ids)} test conflicts: {sample_ids}")
    print()
