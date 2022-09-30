import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def extract_coord_from_path(path):
    '''
    Path expected to look like /path/to/dir/x_y.png
    '''
    x_y = path.stem
    x, y = x_y.split('_')[0], x_y.split('_')[1]
    return int(x), int(y)


def update_state_dict(model_dict, state_dict):
    success, failure = 0, 0
    updated_state_dict = {}
    for k,v in zip(model_dict.keys(), state_dict.values()):
        if v.size() != model_dict[k].size():
            updated_state_dict[k] = model_dict[k]
            failure += 1
        else:
            updated_state_dict[k] = v
            success += 1
    msg = f'{success} weight(s) loaded succesfully ; {failure} weight(s) not loaded because of mismatching shapes'
    return updated_state_dict, msg

def create_train_tune_test_df(
    df: pd.DataFrame,
    save_csv: bool = False,
    output_dir: Path = Path(''),
    tune_size: float = .4,
    test_size: float = .2,
    seed: int = 21,
    ):
    train_df, tune_df = train_test_split(df, test_size=tune_size, random_state=seed)
    train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=seed)
    if save_csv:
        train_df.to_csv(Path(output_dir, f'train.csv'), index=False)
        tune_df.to_csv(Path(output_dir, f'tune.csv'), index=False)
        test_df.to_csv(Path(output_dir, f'test.csv'), index=False)
    train_df = train_df.reset_index()
    tune_df = tune_df.reset_index()
    test_df = test_df.reset_index()
    return train_df, tune_df, test_df