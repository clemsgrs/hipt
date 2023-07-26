import tqdm
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from typing import Optional
from sklearn.neighbors import NearestNeighbors

from source.wsi import WholeSlideImage


def add_random_noise(feature, gamma: float = 0.5, mean: int = 0, std: float = 1.):
    noise = torch.normal(mean, std, size=feature.shape)[0]
    augm_feature = feature + gamma * noise
    return augm_feature


def interpolate_feature(ref_feature, neighbor_feature, lmbda: float = 0.5):
    augm_feature = lmbda * (neighbor_feature - ref_feature) + ref_feature
    return augm_feature


def extrapolate_feature(ref_feature, neighbor_feature, lmbda: float = 0.5):
    augm_feature = lmbda * (ref_feature - neighbor_feature) + ref_feature
    return augm_feature


def get_knn_features(feature, slide_id, region_df, label_df, K: int = 10, sim_threshold: Optional[float] = None):
    label = label_df[label_df.slide_id == slide_id].label.values[0]
    df = pd.merge(region_df, label_df[['slide_id', 'label']], on='slide_id', how='inner')
    # grab all samples belonging to same class
    in_class_df = df[df.label == label].reset_index(drop=True)
    # load features
    features = []
    in_class_feature_paths = list(in_class_df.feature_path.unique())
    with tqdm.tqdm(
        in_class_feature_paths,
        desc='Loading in class features',
        unit=' region',
    ) as t:
        for fp in t:
            f = torch.load(fp)
            features.append(f)
    stacked_features = torch.cat(features, dim=0).numpy()
    knn = NearestNeighbors(
        n_neighbors=K+1,
        metric='cosine',
    )
    knn.fit(stacked_features)
    # retrieve K nearest neighbors
    distances, indices = knn.kneighbors(feature.numpy().reshape(1, -1), return_distance=True)
    # drop the first result which corresponds to the input feature
    distances, indices = distances.squeeze()[1:], indices.squeeze()[1:]
    # cosine distance is defined as (1 - cosine_similarity)
    similarities = (1 - distances)
    # optional thresholding
    if sim_threshold:
        idx = (similarities > sim_threshold).nonzero()[0]
        indices = indices[idx]
        similarities = similarities[idx]
    knn_features = stacked_features[indices]
    knn_df = in_class_df[indices]
    return knn_features, knn_df


def random_augmentation(features, gamma: float = 0.5, mean: int = 0, std: float = 1.):
    augm_features = add_random_noise(features, gamma, mean, std)
    return augm_features


def simple_augmentation(features, slide_id, region_df, label_df, method: str, K: int = 10, sim_threshold: Optional[float] = None, lmbda: float = 0.5):
    augm_features = []
    for feature in features:
        knn_features, knn_df = get_knn_features(feature, slide_id, region_df, label_df, K, sim_threshold)
        # pick a random neighbor, compute augmented feature
        i = random.randint(0,K-1)
        neighbor_feature = knn_features[i]
        if method == "interpolation":
            augm_feature = interpolate_feature(feature, neighbor_feature, lmbda=lmbda)
        elif method == "extrapolation":
            augm_feature = extrapolate_feature(feature, neighbor_feature, lmbda=lmbda)
        else:
            raise KeyError(f"provided method '{method}' not suported ; chose among ['interpolation', 'extrapolation']")
        augm_features.append(augm_feature.unsqueeze(0))
    stacked_augm_features = torch.cat(augm_features, dim=0)
    return stacked_augm_features


def plot_knn_features(feature, x, y, slide_id, region_df, label_df, K: int = 10, sim_threshold: Optional[float] = None, region_dir: Optional[str] = None, slide_dir: Optional[str] = None, spacing: Optional[float] = None, backend: str = 'openslide', size: int = 256, region_fmt: str = 'jpg', dpi: int = 300):
    _, knn_df = get_knn_features(feature, slide_id, region_df, label_df, K, sim_threshold)
    fig, ax = plt.subplots(1, K+1, dpi=dpi)

    # get reference region
    if region_dir:
        fname = Path(region_dir, sid, 'imgs', f'{x}_{y}.{region_fmt}')
        ref_region = Image.open(fname)
    elif slide_dir:
        slide_path = [x for x in slide_dir.glob(f'{sid}*')][0]
        wsi_object = WholeSlideImage(slide_path, backend=backend)
        s = wsi_object.spacing_mapping[spacing]
        ref_region = wsi_object.wsi.get_patch(x, y, ts, ts, spacing=s, center=False)
        ref_region = Image.fromarray(region).convert("RGB")
    else:
        raise ValueError("neither 'region_dir' nor 'slide_dir' was given ; at least one of them must be given")
    ref_region = ref_region.resize((size,size))
    ax[0].imshow(ref_region)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(f'reference', size='xx-small')
    
    # iterate over knn regions
    for i, (_, sid, ts, x, y, sim) in knn_df.iterrows():
        if region_dir:
            fname = Path(region_dir, sid, 'imgs', f'{x}_{y}.jpg')
            region = Image.open(fname)
        elif slide_dir:
            slide_path = [x for x in slide_dir.glob(f'{sid}*')][0]
            wsi_object = WholeSlideImage(slide_path, backend=backend)
            s = wsi_object.spacing_mapping[spacing]
            region = wsi_object.wsi.get_patch(x, y, ts, ts, spacing=s, center=False)
            region = Image.fromarray(region).convert("RGB")
        region = region.resize((size,size))
        ax[i].imshow(region)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f'sim: {sim:.3f}', size='xx-small')
    plt.tight_layout()
    return fig
