import os
import json
import subprocess

def create_and_run_evaluation_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Brain Tumor MRI Classification - Model Evaluation and Comparison\n",
                    "\n",
                    "This notebook loads the training and testing metrics for our three trained models:\n",
                    "- Custom CNN\n",
                    "- MobileNetV2 (Transfer Learning)\n",
                    "- ResNet50 (Transfer Learning)\n",
                    "\n",
                    "It generates comparative visualizations (accuracies, confusion matrices, F1-scores) and provides a final model recommendation.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pickle\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Metrics Data\n",
                    "We read the saved evaluation outputs from the `results/` folder."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "results_dir = '../results'\n",
                    "with open(os.path.join(results_dir, 'custom_cnn_results.pkl'), 'rb') as f:\n",
                    "    cnn_res = pickle.load(f)\n",
                    "with open(os.path.join(results_dir, 'mobilenetv2_results.pkl'), 'rb') as f:\n",
                    "    mnet_res = pickle.load(f)\n",
                    "with open(os.path.join(results_dir, 'resnet50_results.pkl'), 'rb') as f:\n",
                    "    resnet_res = pickle.load(f)\n",
                    "\n",
                    "print(\"Metrics loaded successfully!\")\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Test Accuracy Comparison\n",
                    "We compile the overall testing accuracies and plot a comparative bar chart."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_names = ['Custom CNN', 'MobileNetV2', 'ResNet50']\n",
                    "accuracies = [cnn_res['test_acc'], mnet_res['test_acc'], resnet_res['test_acc']]\n",
                    "\n",
                    "acc_df = pd.DataFrame({\n",
                    "    'Model': model_names,\n",
                    "    'Test Accuracy': accuracies\n",
                    "})\n",
                    "\n",
                    "print(acc_df.to_string(index=False))\n",
                    "\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.barplot(data=acc_df, x='Model', y='Test Accuracy', palette='viridis')\n",
                    "plt.title('Model Test Accuracy Comparison')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.ylim(0, 1.0)\n",
                    "for i, acc in enumerate(accuracies):\n",
                    "    plt.text(i, acc + 0.02, f\"{acc*100:.2f}%\", ha='center', fontweight='bold')\n",
                    "\n",
                    "plt.savefig('../results/model_accuracy_comparison.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Side-by-Side Confusion Matrices\n",
                    "We plot confusion matrices side-by-side to understand error profiles (e.g. glioma confused with meningioma)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                    "\n",
                    "# Custom CNN\n",
                    "sns.heatmap(np.array(cnn_res['confusion_matrix']), annot=True, fmt='d', cmap='Blues', \n",
                    "            xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)\n",
                    "axes[0].set_title(f\"Custom CNN (Acc: {cnn_res['test_acc']*100:.1f}%)\")\n",
                    "axes[0].set_xlabel('Predicted')\n",
                    "axes[0].set_ylabel('True')\n",
                    "\n",
                    "# MobileNetV2\n",
                    "sns.heatmap(np.array(mnet_res['confusion_matrix']), annot=True, fmt='d', cmap='Oranges', \n",
                    "            xticklabels=classes, yticklabels=classes, ax=axes[1], cbar=False)\n",
                    "axes[1].set_title(f\"MobileNetV2 (Acc: {mnet_res['test_acc']*100:.1f}%)\")\n",
                    "axes[1].set_xlabel('Predicted')\n",
                    "axes[1].set_ylabel('True')\n",
                    "\n",
                    "# ResNet50\n",
                    "sns.heatmap(np.array(resnet_res['confusion_matrix']), annot=True, fmt='d', cmap='Greens', \n",
                    "            xticklabels=classes, yticklabels=classes, ax=axes[2], cbar=False)\n",
                    "axes[2].set_title(f\"ResNet50 (Acc: {resnet_res['test_acc']*100:.1f}%)\")\n",
                    "axes[2].set_xlabel('Predicted')\n",
                    "axes[2].set_ylabel('True')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('../results/confusion_matrices_comparison.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Class-Level Metric Analysis (F1-Scores)\n",
                    "We break down performance by class to verify if the models generalized well across the pituitary minority class."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics_list = []\n",
                    "\n",
                    "for name, res in zip(model_names, [cnn_res, mnet_res, resnet_res]):\n",
                    "    rep = res['report']\n",
                    "    for cls in classes:\n",
                    "        cls_key = cls.lower().replace(' ', '')\n",
                    "        # Handle key variations if folder is named differently\n",
                    "        actual_key = cls_key if cls_key in rep else (cls.lower() if cls.lower() in rep else None)\n",
                    "        if actual_key:\n",
                    "            metrics_list.append({\n",
                    "                'Model': name,\n",
                    "                'Class': cls,\n",
                    "                'Precision': rep[actual_key]['precision'],\n",
                    "                'Recall': rep[actual_key]['recall'],\n",
                    "                'F1-Score': rep[actual_key]['f1-score']\n",
                    "            })\n",
                    "\n",
                    "metrics_df = pd.DataFrame(metrics_list)\n",
                    "\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.barplot(data=metrics_df, x='Class', y='F1-Score', hue='Model', palette='Set2')\n",
                    "plt.title('F1-Score Comparison per Tumor Class')\n",
                    "plt.ylabel('F1-Score')\n",
                    "plt.ylim(0, 1.0)\n",
                    "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "plt.savefig('../results/class_f1_comparison.png', bbox_inches='tight', dpi=150)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Classification Reports Text"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for name, res in zip(model_names, [cnn_res, mnet_res, resnet_res]):\n",
                    "    print(f\"\\n=========================================\")\n",
                    "    print(f\" Classification Report: {name}\")\n",
                    "    print(f\"\\n=========================================\")\n",
                    "    print(res['report_text'])\n"
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
    notebook_path = 'notebooks/evaluation.ipynb'
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    print(f"Created {notebook_path} structure.")

    # Run the notebook via nbconvert
    print("Executing notebooks/evaluation.ipynb...")
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
        print("Successfully executed notebooks/evaluation.ipynb and stored outputs.")
    else:
        print("Error executing notebook:")
        print(result.stdout)
        print(result.stderr)

if __name__ == '__main__':
    create_and_run_evaluation_notebook()
