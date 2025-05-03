import os
import glob
import time
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm, trange

# ------------------- Configuration ------------------- #
batch_size = 128
n_epoch = 100
sigma = 25
learning_rate = 1e-3
save_dir = "models"
train_data_dir = "BSDS300/images/train"
test_data_dir = "BSDS300/images/test"
patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(save_dir, exist_ok=True)


# ------------------- CBAM Modules ------------------- #
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pooled = F.adaptive_avg_pool2d(x, 1)
            elif pool_type == "max":
                pooled = F.adaptive_max_pool2d(x, 1)
            if pooled is not None:
                att_raw = self.mlp(pooled)
                channel_att_sum = (
                    att_raw if channel_att_sum is None else channel_att_sum + att_raw
                )

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# ------------------- Data Augmentation ------------------- #
def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    patches = []
    for s in scales:
        h_s, w_s = int(h * s), int(w * s)
        img_s = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_CUBIC)
        for i in range(0, h_s - patch_size + 1, stride):
            for j in range(0, w_s - patch_size + 1, stride):
                x = img_s[i : i + patch_size, j : j + patch_size, :]
                for _ in range(aug_times):
                    patches.append(data_aug(x, mode=np.random.randint(0, 8)))
    return patches


def datagenerator(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.jpg"))
    data = []
    for idx, f in enumerate(files):
        patches = gen_patches(f)
        data += [p for p in patches if p.shape == (patch_size, patch_size, 3)]
        if (idx + 1) % 10 == 0:
            print(f"{idx + 1}/{len(files)} images processed")
    data = np.stack(data, axis=0)
    discard = len(data) - len(data) // batch_size * batch_size
    if discard > 0:
        data = data[:-discard]
    print("Finished generating training data:", data.shape)
    return data


# ------------------- Dataset ------------------- #
class DenoisingDataset(Dataset):
    def __init__(self, xs, sigma):
        self.xs = xs
        self.sigma = sigma

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        clean = self.xs[idx]
        noise = torch.randn_like(clean).mul_(self.sigma / 255.0)
        noisy = clean + noise
        return noisy, clean


# ------------------- REDNet30 with CBAM ------------------- #
class REDNet30_CBAM(nn.Module):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNet30_CBAM, self).__init__()
        self.num_layers = num_layers
        # Encoder conv layers
        self.conv_layers = nn.ModuleList()
        self.cbam_enc = nn.ModuleList()
        # First downsample
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        self.cbam_enc.append(CBAM(num_features))
        # Remaining encoder
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            self.cbam_enc.append(CBAM(num_features))
        # Decoder deconv layers
        self.deconv_layers = nn.ModuleList()
        self.cbam_dec = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        num_features, num_features, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                )
            )
            self.cbam_dec.append(CBAM(num_features))
        # Final upsample
        self.deconv_layers.append(
            nn.ConvTranspose2d(
                num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )
        # No CBAM on final output
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # Encoder
        feats = []
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = self.cbam_enc[i](x)
            # store for skip connection every second layer
            if (i + 1) % 2 == 0 and len(feats) < math.ceil(self.num_layers / 2) - 1:
                feats.append(x)
        # Decoder
        f_idx = 0
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if i < len(self.cbam_dec):
                x = self.cbam_dec[i](x)
            # add skip connections
            if (i + self.num_layers + 1) % 2 == 0 and f_idx < len(feats):
                x = x + feats[-(f_idx + 1)]
                x = self.relu(x)
                f_idx += 1
        x = x + residual
        x = self.relu(x)
        return x


# ------------------- Denoising Function ------------------- #
def denoise_image(img, model):
    img = img.astype("float32") / 255.0
    tensor = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    out = out.squeeze().cpu().numpy().transpose((1, 2, 0))
    out = np.clip(out, 0, 1)
    return (out * 255.0).astype("uint8")


# ------------------- Training + Inference ------------------- #
if __name__ == "__main__":
    model = REDNet30_CBAM().to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.2)

    xs = datagenerator(train_data_dir)
    xs = xs.astype("float32") / 255.0
    xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))
    dataset = DenoisingDataset(xs, sigma)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for epoch in trange(n_epoch):
        model.train()
        epoch_loss = 0
        psnr_acc, ssim_acc = 0, 0
        start = time.time()
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean) / batch_size
            loss.backward()
            optimizer.step()

            clean_np = clean.cpu().numpy()
            out_np = output.detach().cpu().numpy()
            psnr_acc += peak_signal_noise_ratio(clean_np, out_np, data_range=1.0)
            ssim_acc += structural_similarity(
                clean_np.transpose(0, 2, 3, 1),
                out_np.transpose(0, 2, 3, 1),
                channel_axis=-1,
                data_range=1.0,
            )
            epoch_loss += loss.item()

        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{n_epoch}, Loss: {epoch_loss / len(loader):.6f}, PSNR: {psnr_acc / len(loader):.2f}, SSIM: {ssim_acc / len(loader):.4f}, Time: {time.time() - start:.2f}s"
        )
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"rednet_cbam_epoch_{epoch + 1}.pth"),
            )

    # Inference
    model.load_state_dict(
        torch.load(
            os.path.join(save_dir, "rednet_cbam_epoch_100.pth"), map_location=device
        )
    )
    model.eval()
    test_files = glob.glob(os.path.join(test_data_dir, "*.jpg"))
    out_dir = "denoised_results_rednet_cbam"
    os.makedirs(out_dir, exist_ok=True)
    for fp in tqdm(test_files, desc="Testing images"):
        fn = os.path.basename(fp)
        img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        noise = np.random.randn(*img.shape) * sigma
        noisy_img = np.clip(img + noise, 0, 255).astype("uint8")
        denoised = denoise_image(noisy_img, model)
        comp = np.hstack((img, noisy_img, denoised))
        cv2.imwrite(os.path.join(out_dir, fn), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"Saved denoised images to {out_dir}/")
