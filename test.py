import argparse
import torch
import os
import csv
import numpy as np
from dataset import AradDataset
from models.mine import SPAN
from metrics import evaluate_metrics
from torch.utils.data import DataLoader

# === Argomenti da terminale
parser = argparse.ArgumentParser(description='Super Resolution Test')
parser.add_argument('--model', default='', type=str, help='model path')
parser.add_argument('--results', default='', type=str, help='results path')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help="Device to run the script on: 'cuda' or 'cpu'.")
parser.add_argument('--data_path', type=str, default='', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size per forward pass')
opt = parser.parse_args()

print(opt)

# === Carica modello
model = SPAN(31, 31)
model.load_state_dict(torch.load(opt.model, weights_only=True))
model = model.to(opt.device)
model.eval()

# === Dataset e DataLoader
test_set = AradDataset(opt.data_path, train=False)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)

# === Contatori metriche
sam_total, psnr_total, ssim_total = 0.0, 0.0, 0.0
all_metrics = []

print('===> Testing')
for iteration, (X, gt) in enumerate(test_loader):
    X = X.to(opt.device)
    gt = gt.to(opt.device)

    with torch.no_grad():
        pred = model(X)

    pred_np = pred.squeeze(0).cpu().numpy()  # [C, H, W]
    gt_np = gt.squeeze(0).cpu().numpy()      # [C, H, W]

    metrics = evaluate_metrics(pred_np, gt_np)
    all_metrics.append((iteration, metrics['SAM'], metrics['PSNR'], metrics['SSIM']))

    sam_total += metrics['SAM']
    psnr_total += metrics['PSNR']
    ssim_total += metrics['SSIM']

    print(f"\n===== Image {iteration} =====")

# === Medie finali
avg_sam = sam_total / len(test_loader)
avg_psnr = psnr_total / len(test_loader)
avg_ssim = ssim_total / len(test_loader)

print("\n=====> AVERAGE RESULTS")
print(f"Avg. SAM : {avg_sam:.4f}")
print(f"Avg. PSNR: {avg_psnr:.2f}")
print(f"Avg. SSIM: {avg_ssim:.4f}")

# === Scrittura su file CSV
results_file = opt.results
with open(results_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'SAM', 'PSNR', 'SSIM'])

    for img_idx, sam, psnr, ssim in all_metrics:
        writer.writerow([
            img_idx,
            f"{sam:.4f}",
            f"{psnr:.2f}",
            f"{ssim:.4f}"
        ])

    writer.writerow([])
    writer.writerow(['AVG', f"{avg_sam:.4f}", f"{avg_psnr:.2f}", f"{avg_ssim:.4f}"])

print(f"\n📁 Risultati salvati in: {results_file}")

