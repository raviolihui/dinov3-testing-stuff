import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dinov3.models.vision_transformer import vit_large


# Configuration parameters for hyperbolic tests
IMG_DIR = "/home/carmenoliver/my_projects/processed_images"
BATCH_SIZE = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16
HYPO_DIM = 128
C = 1.0  # curvature
NUM_POS = 3  # positives per image

#DATASET
class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                          if f.lower().endswith((".jpg", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# --- Transform ---
transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
])

#MODEL COMPONENTS
class ViTFeatureExtractor(nn.Module):
    def __init__(self, patch_size=16, pretrained_path=None):
        super().__init__()
        self.model = vit_large(patch_size=patch_size, num_register_tokens=0)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)

class HyperbolicHead(nn.Module):
    def __init__(self, in_dim, out_dim, c=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.c = c

    def exp_map(self, x):
        device = x.device
        norm = torch.norm(x, dim=-1, keepdim=True)
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=device))
        return torch.tanh(sqrt_c * norm) * x / (norm + 1e-5)

    def forward(self, x):
        x = self.linear(x)
        return self.exp_map(x)


#Poincare operations

def mobius_add(u, v, c=1.0):
    u2 = (u**2).sum(dim=-1, keepdim=True)
    v2 = (v**2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    numerator = (1 + 2*c*uv + c*v2)*u + (1 - c*u2)*v
    denominator = 1 + 2*c*uv + c**2 * u2 * v2
    return numerator / (denominator + 1e-5)

def poincare_distance(u, v, c=1.0):
    diff = mobius_add(-u, v, c)
    norm = diff.norm(dim=-1, keepdim=True)
    return 2 / torch.sqrt(torch.tensor(c, device=u.device)) * \
           torch.atanh(torch.clamp(torch.sqrt(torch.tensor(c, device=u.device)) * norm, max=1-1e-5))

def hyperbolic_contrastive_loss(hyp_emb, pos_pairs, neg_pairs=None, margin=1.0):
    # pos_pairs: tensor of shape [num_pairs, 2]
    i, j = pos_pairs[:,0], pos_pairs[:,1]
    d_pos = poincare_distance(hyp_emb[i], hyp_emb[j])
    loss = (d_pos**2).mean()
    if neg_pairs is not None:
        i, j = neg_pairs[:,0], neg_pairs[:,1]
        d_neg = poincare_distance(hyp_emb[i], hyp_emb[j])
        loss += (F.relu(margin - d_neg)**2).mean()
    return loss

def train_hyperbolic_head(features, hyp_head, pos_pairs, lr=1e-3):
    hyp_head.train()
    optimizer = torch.optim.Adam(hyp_head.parameters(), lr=lr)
    optimizer.zero_grad()
    hyp_emb = hyp_head(features)
    loss = hyperbolic_contrastive_loss(hyp_emb, pos_pairs)
    loss.backward()
    optimizer.step()
    return hyp_emb, loss.item()

# Visualization functions

def plot_pca_2d(embeddings, title="2D PCA"):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(6,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1])
    plt.title(title)
    plt.grid(True)
    plt.show()
    return emb_2d

def plot_poincare_disk(embeddings, title="Hyperbolic Embeddings"):
    norms = torch.norm(torch.tensor(embeddings), dim=-1, keepdim=True)
    scaled = embeddings / (norms * 1.1)
    fig, ax = plt.subplots(figsize=(6,6))
    circle = plt.Circle((0,0), 1, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)
    plt.scatter(scaled[:,0], scaled[:,1])
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()
ç

#Code Script 

dataset = ImageFolderDataset(IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Load Model ---
vit_extractor = ViTFeatureExtractor(pretrained_path="/home/carmenoliver/my_projects/dynov3-testing-stuff/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth").to(DEVICE)

# --- Get One Batch ---
image_batch = next(iter(dataloader)).to(DEVICE)
features = vit_extractor(image_batch)

print(f"Feature shape: {features.shape}")

# --- Hyperbolic Head ---
hyp_head = HyperbolicHead(features.shape[1], HYPO_DIM, c=C).to(DEVICE)
hyp_emb = hyp_head(features)

# --- Positive pairs ---
sims = cosine_similarity(features.cpu())
topk_neighbors = np.argsort(-sims, axis=1)[:, 1:NUM_POS+1]
i_idx = np.repeat(np.arange(sims.shape[0]), NUM_POS)
j_idx = topk_neighbors.flatten()
pos_pairs = torch.tensor(list(zip(i_idx, j_idx)), dtype=torch.long)

# --- Train Hyperbolic Head ---
hyp_emb, loss_val = train_hyperbolic_head(features, hyp_head, pos_pairs)
print("Hyperbolic contrastive loss:", loss_val)

# --- Visualization ---
emb_2d = plot_pca_2d(features.detach().cpu().numpy(), "Euclidean Features")
plot_poincare_disk(hyp_emb.detach().cpu().numpy(), "Hyperbolic Embeddings")

# --- Show Nearest Neighbors ---
query_idx = 0
dists = poincare_distance(hyp_emb[query_idx:query_idx+1], hyp_emb).squeeze().detach().cpu().numpy()
topk_hyp = dists.argsort()[:5]
show_selected_images(image_batch, [query_idx]+[i for i in topk_hyp if i != query_idx], title="Query + Hyperbolic Neighbors")