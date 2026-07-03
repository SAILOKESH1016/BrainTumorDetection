# Data Dictionary: Brain Tumor MRI Dataset

This document details the features, shapes, folders, and encoding mappings of the Brain Tumor MRI classification project.

## Class Encodings & Labels

The prediction engine targets four distinct mutually-exclusive classes. The integer encodings are mapped alphabetically based on folder names:

| Folder Name | Label Name | Numeric Encoding | Description |
|---|---|---|---|
| `glioma` | Glioma | `0` | Tumor originating in the glial cells of the brain or spine. |
| `meningioma` | Meningioma | `1` | Typically benign tumor originating in the meninges. |
| `notumor` | No Tumor | `2` | Control scans representing normal, tumor-free brain structure. |
| `pituitary` | Pituitary | `3` | Tumor originating in the pituitary gland at the base of the brain. |

---

## File Attributes

Each data point is an individual image file.

| Attribute | Type | Values / Units | Description |
|---|---|---|---|
| `filepath` | string | e.g. `archive/Training/glioma/Tr-gl_0010.jpg` | Absolute or relative path to the image on disk. |
| `filename` | string | e.g. `Tr-gl_0010.jpg` | Base name of the file on disk. |
| `label` | string | `glioma`, `meningioma`, `notumor`, `pituitary` | Target classification label of the image. |
| `class_id` | integer | `0`, `1`, `2`, `3` | Integer index mapped to the target label. |
| `width` | integer | Pixels (typically 150 to 512+) | Horizontal dimension of the raw image. |
| `height` | integer | Pixels (typically 150 to 512+) | Vertical dimension of the raw image. |
| `channels` | integer | `1` or `3` | Channels (scans are typically single-channel grayscale but saved in RGB format). |
| `format` | string | `JPEG` | The compression format of the image files. |
| `split` | string | `train` or `test` or `val` | Partition to which the image is assigned. |
