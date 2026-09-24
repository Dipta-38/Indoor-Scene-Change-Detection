## Dataset Preparation


```text
Dataset Directory Structure
data/
│
├── raw/
│ ├── images/
│ └── labels/
│
├── preprocessed/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ │
│ ├── validation/
│ │ ├── images/
│ │ └── labels/
│ │
│ └── test/
│ ├── images/
│ └── labels/
│
└── data.yaml
└── classes.txt

```

**Add**, **Delete**, and **Move** samples are retained for the change-detection experiment. Files are parsed into before/after reference-query pairs using their filename metadata. Invalid or ambiguous pair IDs are excluded before splitting.

A pair-level split is used so both images belonging to the same scene-change pair remain in the same partition.

| Split | Add | Delete | Move | Total |
|---|---:|---:|---:|---:|
| Train | 232 | 199 | 209 | **640** |
| Validation | 50 | 43 | 44 | **137** |
| Test | 50 | 43 | 45 | **138** |
| **Total** | **332** | **285** | **298** | **915** |

<p align="center">
  <img src="assets/dataset_distribution.png" alt="Dataset distribution" width="95%">
</p>

<details>
<summary><b>Change-type balance by split</b></summary>
<br>
<p align="center">
  <img src="assets/dataset_distribution_by_split.png" alt="Change type balance by split" width="78%">
</p>
</details>
