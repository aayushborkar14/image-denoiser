import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
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
    img = cv2.imread(file_name, 0)
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(
            img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC
        )
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i : i + patch_size, j : j + patch_size]
                for _ in range(aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir):
    file_list = glob.glob(os.path.join(data_dir, "*.jpg"))
    data = []
    for i, file_path in enumerate(file_list):
        patches = gen_patches(file_path)
        for p in patches:
            if p.shape == (patch_size, patch_size):
                data.append(p)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(file_list)} images processed")

    data = np.stack(data, axis=0)
    data = np.expand_dims(data, axis=3)
    discard_n = len(data) - len(data) // batch_size * batch_size
    if discard_n > 0:
        data = np.delete(data, range(discard_n), axis=0)

    print("Finished generating training data:", data.shape)
    return data


# ------------------- Dataset ------------------- #
class DenoisingDataset(Dataset):
    def __init__(self, xs, sigma):
        super().__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        clean = self.xs[index]
        noise = torch.randn_like(clean).mul_(self.sigma / 255.0)
        noisy = clean + noise
        return noisy, clean

    def __len__(self):
        return self.xs.size(0)


# ------------------- Model ------------------- #
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super().__init__()
        padding = 1
        layers = [
            nn.Conv2d(
                image_channels, n_channels, kernel_size, padding=padding, bias=True
            ),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(
                    n_channels, n_channels, kernel_size, padding=padding, bias=False
                ),
                nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
                nn.ReLU(inplace=True),
            ]
        layers.append(
            nn.Conv2d(
                n_channels, image_channels, kernel_size, padding=padding, bias=False
            )
        )
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return x - self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# ------------------- Denoising Function ------------------- #
def denoise_image(img, model):
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    out = out.squeeze().cpu().numpy()
    out = np.clip(out, 0, 1)
    return (out * 255.0).astype("uint8")


# ------------------- Training + Inference ------------------- #
if __name__ == "__main__":
    model = DnCNN().to(device)
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
        psnr_loss, ssim = 0, 0
        start_time = time.time()

        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean) / batch_size
            loss.backward()
            optimizer.step()

            psnr_loss += peak_signal_noise_ratio(
                clean.detach().cpu().numpy(),
                output.detach().cpu().numpy(),
                data_range=1.0,
            )
            ssim += structural_similarity(
                clean.detach().cpu().numpy(),
                output.detach().cpu().numpy(),
                data_range=1.0,
                channel_axis=1,
            )

            epoch_loss += loss.item()

        scheduler.step()
        psnr_loss /= len(loader)
        ssim /= len(loader)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{n_epoch}, Loss: {epoch_loss / len(loader):.6f}, Time: {elapsed:.2f}s"
        )
        print(f"PSNR: {psnr_loss:.2f}, SSIM: {ssim:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"dncnn_epoch_{epoch + 1}.pth"),
            )

    # --- Inference on test set after training --- #
    model_path = os.path.join(save_dir, "dncnn_epoch_100.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_files = glob.glob(os.path.join(test_data_dir, "*.jpg"))
    output_dir = "denoised_results"
    os.makedirs(output_dir, exist_ok=True)

    for file_path in tqdm(test_files, desc="Testing images"):
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path, 0)
        noise = np.random.randn(*img.shape) * sigma
        noisy_img = np.clip(img + noise, 0, 255).astype("uint8")
        denoised_img = denoise_image(noisy_img, model)
        comparison = np.hstack((img, noisy_img, denoised_img))
        cv2.imwrite(os.path.join(output_dir, filename), comparison)

    print(f"Saved denoised images to {output_dir}/")
