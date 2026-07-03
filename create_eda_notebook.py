import os
import json
import subprocess

def create_and_run_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Brain Tumor MRI Classification - Exploratory Data Analysis\n",
                    "\n",
                    "This notebook performs Exploratory Data Analysis (EDA) on the brain MRI scans. We analyze:\n",
                    "- Dataset splits and class balances\n",
                    "- Resolution distributions\n",
                    "- Visual sample grids for each tumor class\n",
                    "- Average intensity heatmaps per class\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "import os\n",
                    "sys.path.insert(0, os.path.abspath('..'))\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from PIL import Image\n",
                    "import cv2\n",
                    "from preprocessing import prepare_splits\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Dataset Splits\n",
                    "We use our preprocessing pipeline to scan the `archive/` folder and prepare the clean splits."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_df, val_df, test_df, le = prepare_splits('../archive')\n",
                    "train_df['split'] = 'Train'\n",
                    "val_df['split'] = 'Validation'\n",
                    "test_df['split'] = 'Test'\n",
                    "full_df = pd.concat([train_df, val_df, test_df])\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Class Distribution Analysis\n",
                    "We analyze the count of images in each of the four categories across the splits to ensure they are representative."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "os.makedirs('results', exist_ok=True)\n",
                    "\n",
                    "# Class count plot\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.countplot(data=full_df, x='label', hue='split', palette='viridis')\n",
                    "plt.title('Class Distribution Across Splits')\n",
                    "plt.xlabel('Tumor Category')\n",
                    "plt.ylabel('Image Count')\n",
                    "plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
                    "plt.savefig('results/class_distribution.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n",
                    "\n",
                    "# Table of distributions\n",
                    "dist_table = full_df.groupby(['label', 'split']).size().unstack(fill_value=0)\n",
                    "dist_table['Total'] = dist_table.sum(axis=1)\n",
                    "print(dist_table)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Image Dimension Analysis\n",
                    "We analyze image dimensions to see the variation in resolutions across scans."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "resolutions = []\n",
                    "# Sample up to 300 images to inspect shape\n",
                    "for idx, row in full_df.sample(min(300, len(full_df)), random_state=42).iterrows():\n",
                    "    try:\n",
                    "        with Image.open(row['filepath']) as img:\n",
                    "            w, h = img.size\n",
                    "            resolutions.append((w, h))\n",
                    "    except Exception:\n",
                    "        pass\n",
                    "\n",
                    "resolutions = np.array(resolutions)\n",
                    "print(f\"Minimum resolution (width, height): ({resolutions[:,0].min()}, {resolutions[:,1].min()})\")\n",
                    "print(f\"Maximum resolution (width, height): ({resolutions[:,0].max()}, {resolutions[:,1].max()})\")\n",
                    "print(f\"Mean resolution (width, height): ({resolutions[:,0].mean():.1f}, {resolutions[:,1].mean():.1f})\")\n",
                    "\n",
                    "plt.figure(figsize=(8, 6))\n",
                    "sns.scatterplot(x=resolutions[:,0], y=resolutions[:,1], alpha=0.6, color='purple')\n",
                    "plt.title('Image Resolutions Distribution')\n",
                    "plt.xlabel('Width (pixels)')\n",
                    "plt.ylabel('Height (pixels)')\n",
                    "plt.grid(True)\n",
                    "plt.savefig('results/resolution_scatter.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visualize Sample MRI Scans\n",
                    "We plot a grid of sample images from each tumor category to understand the visual differences."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "classes = full_df['label'].unique()\n",
                    "fig, axes = plt.subplots(4, 4, figsize=(12, 12))\n",
                    "\n",
                    "for i, cls in enumerate(classes):\n",
                    "    cls_df = full_df[full_df['label'] == cls]\n",
                    "    samples = cls_df.sample(4, random_state=42)\n",
                    "    for j, (_, row) in enumerate(samples.iterrows()):\n",
                    "        img = Image.open(row['filepath'])\n",
                    "        ax = axes[i, j]\n",
                    "        ax.imshow(img, cmap='gray' if len(img.getbands()) == 1 else None)\n",
                    "        ax.set_title(f\"{row['label'].capitalize()}\\n{img.size}\", fontsize=10)\n",
                    "        ax.axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('results/sample_images.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Average Intensity Heatmaps\n",
                    "We compute the average image intensity profile for each class. This shows whether there are class-specific location or size biases in the MRI slices."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(14, 8))\n",
                    "for i, cls in enumerate(classes):\n",
                    "    cls_df = full_df[full_df['label'] == cls]\n",
                    "    avg_img = np.zeros((224, 224), dtype=np.float32)\n",
                    "    count = 0\n",
                    "    \n",
                    "    # Average over 100 samples per class\n",
                    "    for _, row in cls_df.sample(min(100, len(cls_df)), random_state=42).iterrows():\n",
                    "        img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)\n",
                    "        if img is not None:\n",
                    "            img_res = cv2.resize(img, (224, 224))\n",
                    "            avg_img += img_res / 255.0\n",
                    "            count += 1\n",
                    "            \n",
                    "    if count > 0:\n",
                    "        avg_img /= count\n",
                    "        \n",
                    "    plt.subplot(2, 2, i+1)\n",
                    "    sns.heatmap(avg_img, cmap='inferno', cbar=False)\n",
                    "    plt.title(f\"Average Intensity - {cls.capitalize()} (n={count})\")\n",
                    "    plt.axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('results/average_intensity_maps.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs('notebooks', exist_ok=True)
    notebook_path = 'notebooks/eda.ipynb'
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Created {notebook_path} structure.")

    # Run the notebook via nbconvert
    print("Executing notebooks/eda.ipynb...")
    cmd = [
        r".venv\Scripts\jupyter",
        "nbconvert",
        "--to", "notebook",
        "--execute",
        notebook_path,
        "--inplace"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Successfully executed notebooks/eda.ipynb and stored outputs.")
    else:
        print("Error executing notebook:")
        print(result.stdout)
        print(result.stderr)

if __name__ == '__main__':
    create_and_run_notebook()
