import shutil
import os
from pathlib import Path

def move_bb_preds(source_path, target_path):
    for set in source_path.iterdir():
        for city in set.iterdir():
            t_dir = target_path / set.stem / city.stem
            shutil.move(str(city / "bb_preds"), str(t_dir / "bb_preds"))

    shutil.rmtree(source_path)

def get_test_set(data_path):
    train_path = data_path / "train"

    for city in train_path.iterdir():
        # print(city)
        if city.stem in ["monchengladbach", "ulm", "darmstadt"]:
            t_dir = data_path / "test" / city.stem
            # t_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(city), str(t_dir))


if __name__ == '__main__':

    processed_path = Path("./processed")
    bb_path = Path("./bb_preds")

    move_bb_preds(bb_path, processed_path)
    get_test_set(processed_path)