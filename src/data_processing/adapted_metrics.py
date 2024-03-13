import os
import zipfile
from glob import glob
from random import random

import numpy as np
from cleanfid.utils import ResizeDataset
from scipy import linalg
from skimage import color
from scipy.stats import wasserstein_distance
import torch
from cleanfid.features import build_feature_extractor, get_reference_statistics
from tqdm import tqdm


def normalize_pop(h1: np.array, step=1):
    return h1 / np.sum(h1) / step

def get_hist(img: np.ndarray) -> np.ndarray:
    """
    Get the histogram of an image.

    :param img: image whose histogram is to be computed.
    :return: image histogram.
    """
    hist = [np.histogram(img[..., j].flatten(), bins=256, range=[0, 256],
                         density=True)[0] for j in range(3)]
    return np.array(hist)

def unpaired_lab_WB(img_set1, img_set2):
    step = 1
    bins = np.arange(-128,128,step)
    chr1 = np.zeros((2, len(bins) - 1))
    chr2 = chr1.copy()
    av_hists = []
    hist_1 = np.zeros((3,256))
    hist_2 = np.zeros((3,256))
    index = -1
    for img1 in img_set1:
        index += 1
        img1 = img1.cpu().numpy()
        img2 = img_set2[index].cpu().numpy()
        hist_1 += get_hist(img1)
        hist_2 += get_hist(img2)
        lab1 = color.rgb2lab(img1)
        lab2 = color.rgb2lab(img2)
        chr1_values1 = np.clip(np.ravel(lab1[:, :, 1]), -128, 127)
        chr2_values1 = np.clip(np.ravel(lab1[:, :, 2]), -128, 127)
        chr1[0] += np.histogram(chr1_values1, bins=bins)[0]
        chr2[0] += np.histogram(chr2_values1, bins=bins)[0]
        chr1_values2 = np.clip(np.ravel(lab2[:, :, 1]), -128, 127)
        chr2_values2 = np.clip(np.ravel(lab2[:, :, 2]), -128, 127)
        chr1[1] += np.histogram(chr1_values2, bins=bins)[0]
        chr2[1] += np.histogram(chr2_values2, bins=bins)[0]
    av_hists.append(hist_1/(index+1))
    av_hists.append(hist_2/(index+1))
    lab_wd = max(
        wasserstein_distance(normalize_pop(chr1[0]), normalize_pop(chr1[1])),
        wasserstein_distance(normalize_pop(chr2[0]), normalize_pop(chr2[1])))
    return lab_wd



def calculate_fid(paths: list, batch_size: int, device: str, dims: int = 512,
                  num_workers: int = 4):

    return compute_fid(paths[0], paths[1], mode="clean",
                           num_workers=num_workers,
                           batch_size=batch_size, device=device, z_dim=dims)
def compute_fid(fdir1=None, fdir2=None, gen=None,
            mode="clean", model_name="inception_v3", num_workers=12,
            batch_size=32, device=torch.device("cuda"), dataset_name="FFHQ",
            dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512,
            custom_feat_extractor=None, verbose=True,
            custom_image_tranform=None, custom_fn_resize=None, use_dataparallel=True):
    if custom_feat_extractor is None and model_name=="inception_v3":
        feat_model = build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
    elif custom_feat_extractor is None and model_name=="clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor
    score = compare_folders(fdir1, fdir2, feat_model,
                            mode=mode, batch_size=batch_size,
                            num_workers=num_workers, device=device,
                            custom_image_tranform=custom_image_tranform,
                            custom_fn_resize=custom_fn_resize,
                            verbose=verbose)
    return score

def compare_folders(fdir1, fdir2, feat_model, mode, num_workers=0,
                    batch_size=8, device=torch.device("cuda"), verbose=True,
                    custom_image_tranform=None, custom_fn_resize=None):
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname1} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_tranform,
                                    custom_fn_resize=custom_fn_resize)
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_tranform,
                                    custom_fn_resize=custom_fn_resize)
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None):
    # get all relevant files in the dataset
    if ".zip" in fdir:
        files = list(set(zipfile.ZipFile(fdir).namelist()))
        # remove the non-image files inside the zip
        files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in EXTENSIONS]
    else:
        files = sorted([file for ext in EXTENSIONS
                    for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device, mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  custom_image_tranform=custom_image_tranform,
                                  description=description, fdir=fdir, verbose=verbose)
    return np_feats

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None,
                       description="", fdir=None, verbose=True,
                       custom_image_tranform=None):
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fdir=fdir, mode=mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats

def get_batch_features(batch, model, device):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()