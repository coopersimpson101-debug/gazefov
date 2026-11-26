# Real IR eye data (⏳ not evaluated yet)

All accuracy numbers in this repository are on **synthetic** images (🧪) because they come with exact
ground truth for gaze in degrees. No real IR eye dataset was downloaded or evaluated for v0.1.0:
during development the candidate dataset page could not be fetched over HTTPS from the build machine
(certificate mismatch), so its licence and size could not be verified, and the repo rules only allow
openly licensed data from an official academic source with a recorded licence.

Candidates to evaluate (check each licence yourself before downloading):

| dataset | what it has | ground truth usable here |
|---|---|---|
| Labelled Pupils in the Wild (LPW), Tonsen et al., ETRA 2016 | 66 near-eye IR videos, 22 people, head-mounted tracker | pupil centre (px) → pupil detection rate / px error |
| Świrski, Bulling & Dodgson, ETRA 2012 | high-off-axis IR frames with hand-labelled pupil ellipses | pupil ellipse → centre / shape error |
| ExCuSe / ElSe (Fuhl et al. 2015, 2016) | challenging real-world IR frames | pupil centre (px) |

None of these provide calibrated LED/camera geometry, so only the **pupil detector** can be scored on
them; the 3D gaze model needs a rig like the one in docs/hardware.md.

## How to run it when you have a dataset

1. Download into `data/<name>/` (gitignored) from the dataset's official page; record the URL,
   licence and size in the table below.
2. Write a loader yielding `(image uint8 HxW, pupil_center_px)`; call
   `gazefov.pupil.detect_pupil(image)` and report detection rate (error < 5 px) and median error.

| dataset | source URL | licence | size | frames scored | detection rate @5 px | median px error |
|---|---|---|---|---:|---:|---:|
| | | | | | | |
