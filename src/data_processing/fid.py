from scipy import linalg
from cleanfid.features import build_feature_extractor
from cleanfid.resize import *

def calculate_fid(images1, images2, batch_size, device=0, dims=256):
    # Convert images to tensors if they are numpy arrays
    if isinstance(images1, np.ndarray):
        images1 = torch.tensor(images1, dtype=torch.float32).to(device)
    if isinstance(images2, np.ndarray):
        images2 = torch.tensor(images2, dtype=torch.float32).to(device)

    return compute_fid(images1, images2, batch_size=batch_size, device=device, z_dim=dims)

def compute_fid(images1, images2, mode="clean", num_workers=12, batch_size=32,
                device=torch.device("cuda"), z_dim=512):
    # Assuming we're using the same feature extractor as before
    feat_model = build_feature_extractor(mode, device)

    # Get features for images1
    feats1 = get_features(images1, feat_model, batch_size, device)
    mu1 = torch.mean(feats1, dim=0)
    sigma1 = torch_cov(feats1)

    # Get features for images2
    feats2 = get_features(images2, feat_model, batch_size, device)
    mu2 = torch.mean(feats2, dim=0)
    sigma2 = torch_cov(feats2)

    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid.item()

def get_features(images, model, batch_size, device):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = model(batch)
        feats.append(feat)
    feats = torch.cat(feats)
    return feats

def torch_cov(feats):
    # Compute covariance matrix
    feats = feats - torch.mean(feats, dim=0, keepdim=True)
    cov = feats.t() @ feats / (len(feats) - 1)
    return cov

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = torch.eye(sigma1.shape[0], device=mu1.device) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    return (diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)

# Now you can use calculate_fid with tensors or ndarrays as inputs.
