# Project Milestones & Verification Criteria

This tracking sheet lists the definition of done and validation checklists for each stage of our deep learning project.

---

## [x] Milestone 1: Planning and Setup (Phase 1)
- [x] Select the Kaggle Brain Tumor MRI Dataset and verify its directory structure in `archive/`.
- [x] Create project implementation guidelines, project plan, milestones tracker.
- [x] Define requirement packages and configure git repository files.

---

## [ ] Milestone 2: Data Preprocessing & Validation (Phase 2)
- [ ] Implement `preprocessing.py` containing modular data loader, validation, scaling, and augmentation.
- [ ] Scan and filter corrupted/empty images.
- [ ] Implement a duplicate image detection process.
- [ ] Partition `archive/Training` into clean Train and Validation folders.

---

## [ ] Milestone 3: Exploratory Data Analysis (Phase 3)
- [ ] Create `notebooks/eda.ipynb` and execute it fully.
- [ ] Document image dimensions, missing files, and class balances.
- [ ] Visualize class counts and sample images.
- [ ] Summarize findings in `eda_report.md`.

---

## [ ] Milestone 4: Model Engineering (Phase 4)
- [ ] Implement `train.py` training script.
- [ ] Define 3 architectures: Custom CNN, MobileNetV2 Transfer Learning, ResNet50 Transfer Learning.
- [ ] Incorporate callbacks: Early Stopping, Learning Rate Scheduler, Checkpoints.
- [ ] Execute CPU-friendly training runs and log training metrics (loss, accuracy).

---

## [ ] Milestone 5: Optimization & Evaluation (Phase 5)
- [ ] Create `optimization.md` highlighting strategies (dropout, learning rate tuning).
- [ ] Create `notebooks/evaluation.ipynb` comparing architectures using Classification Reports (precision, recall, F1) and Confusion Matrices.
- [ ] Export results tables and confusion matrix figures to `results/`.

---

## [ ] Milestone 6: Deployment & Packaging (Phase 6)
- [ ] Package model weights (`.h5`), class label mappings (`.pkl`), and preprocessing configurations (`.pkl`) in `models/`.
- [ ] Build `app.py` Streamlit UI with file uploader, prediction engine, confidence estimator, and EDA tab.
- [ ] Run application locally and verify functional stability on mock images.

---

## [ ] Milestone 7: Documentation & Viva Preparation (Phase 7)
- [ ] Write a complete `README.md` with system overview, architecture, performance, and CLI instructions.
- [ ] Generate comprehensive final project report content (`report/report.md`).
- [ ] Review 20 QA Viva preparation questions in `viva_preparation.md`.
