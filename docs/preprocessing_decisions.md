# Preprocessing Decisions for Egyptian License Plate Recognition

This document explains why each Part 5 transform is included (or
deliberately left disabled by default) in `configs/preprocessing_config.yaml`,
specifically in the context of Egyptian ALPR imagery.

## Context that drives these choices

Egyptian license-plate source imagery — whether from traffic cameras,
dashcams, parking-lot CCTV, or phone captures used in Kaggle datasets —
tends to share a few recurring characteristics:

- **Strong, uneven daylight glare and shadow.** Egypt's climate means
  most captures are in bright, high-contrast sunlight; plates are often
  partially in direct glare and partially in vehicle-body shadow within
  the same crop.
- **Dual-script plates.** Modern Egyptian plates carry both Arabic-indic
  digits/letters and Latin transliteration in the same plate area, so
  preprocessing must preserve fine character-stroke edges for *both*
  scripts rather than optimizing for a single alphabet's stroke width.
- **Mixed capture quality.** Kaggle-sourced datasets pool images from
  many original sources (phone photos, traffic-cam frames, scraped web
  images), so noise levels, compression artifacts, and resolution vary
  widely within one dataset — the pipeline has to be robust to this
  heterogeneity, not tuned to one narrow capture profile.
- **Perspective and rotation variance.** Cameras are rarely perfectly
  perpendicular to the plate, especially in dashcam/CCTV footage,
  producing mild-to-moderate perspective skew and in-plane rotation.

## Enabled by default

### Denoising — `denoise` (Non-Local Means, strength 7.0)
Chosen over Gaussian/median blur as the *first* step because NLM removes
sensor/compression noise while preserving edges far better than local
blurs — important since character-edge sharpness is the single most
predictive feature for downstream OCR accuracy. A moderate strength (7.0)
avoids smoothing away thin Arabic diacritic-like strokes.

### CLAHE — `clahe` (clip_limit 2.0, tile 8x8)
Preferred over global histogram equalization for the glare/shadow problem
described above: CLAHE operates on local tiles, so it lifts shadowed
regions of a plate without blowing out already-bright glared regions in
the same image. Applied on the L channel (LAB space) to avoid introducing
color casts. clip_limit=2.0 is a conservative default that boosts local
contrast without amplifying noise in flat, low-texture background regions.

### Gamma correction — `gamma_correction` (gamma 1.15)
A mild brightening applied *after* CLAHE to lift the many dusk/tunnel/
underpass captures common in Egyptian traffic footage, where the whole
frame — not just part of it — is under-exposed. gamma > 1 brightens;
1.15 was chosen as a gentle default that doesn't wash out already
well-exposed daytime shots (tune upward only if `brightness_histogram.png`
from the EDA stage shows a strong low-brightness cluster).

### Bilateral filter — `bilateral_filter` (d=7, sigma_color=60, sigma_space=60)
Applied after CLAHE/gamma as a second, edge-preserving smoothing pass to
clean up the noise that local contrast enhancement can amplify, without
softening the plate character boundaries that a Gaussian blur of
equivalent strength would blur away. This directly serves OCR legibility.

### Sharpen — `sharpen` (unsharp mask, amount 0.6)
A restrained unsharp-mask pass to counteract the residual softening from
denoising + bilateral filtering, and to compensate for motion blur that's
common in dashcam and moving-vehicle captures. amount=0.6 is deliberately
conservative — aggressive sharpening amplifies JPEG compression artifacts,
which are prevalent in scraped/aggregated Kaggle imagery.

### Letterbox — `letterbox` (aspect-ratio preserving, pad to `target_size`)
Chosen over a plain `resize` for the final sizing step because plain
resize distorts character aspect ratio (a '0' can start looking like an
'O' stretched differently, etc.), which measurably hurts OCR and can also
skew a downstream detector's learned aspect-ratio priors. Letterbox pads
with a neutral gray (114,114,114) — the same convention as YOLO-family
detectors — so it composes cleanly with typical detection/OCR training
recipes without requiring a different padding convention downstream.

## Available but disabled by default

### Perspective correction — `perspective_correction_hook`
Disabled because true perspective correction requires the four plate
corner points, which the *dataset preparation* phase does not itself
produce (that requires either manual annotation or a corner/keypoint
model, which belongs to the modeling phase, not data engineering). The
hook is implemented and wired into the step registry so that once a
corner-detection model exists, it can be dropped in without changing the
pipeline's architecture — only `configs/preprocessing_config.yaml` needs
a `corner_points` param source.

### Rotation correction — `rotation_correction_hook`
Disabled by default because its automatic Hough-line-based angle
estimate is unreliable on plate crops with heavy background clutter
(common in raw, uncropped traffic-camera frames) and can introduce
*more* skew than it removes on those images. Recommended to enable only
after inspecting `bbox_size_distribution.png` / manually spot-checking a
sample once the dataset's actual crop tightness is known; safe to enable
globally once plate crops are tight (e.g., after a detector-based
pre-crop stage).

### Global histogram equalization — `histogram_equalization`
Left disabled (in favor of CLAHE) because global equalization
over-amplifies contrast in the glare-heavy conditions described above —
it tends to make already-bright glared plate regions completely blow out
to white, destroying character strokes exactly where legibility matters
most.

### Brightness / contrast normalization — `brightness_normalization`, `contrast_normalization`
Left disabled because they duplicate what CLAHE + gamma already achieve
via a more locally-aware mechanism; enabling both simultaneously risks
over-correcting mid-tones and reducing effective dynamic range. Kept
available for ablation experiments (e.g., testing simpler baselines
against the CLAHE+gamma combination).

### Gaussian blur / median filter — `gaussian_blur`, `median_filter`
Left disabled by default because denoise (NLM) + bilateral filtering
already provide the noise-reduction this pipeline needs, with better
edge preservation. Adding either would double-smooth thin character
strokes — a real risk for Arabic-script digits, whose distinguishing
features are often small connecting strokes and dots that a second blur
pass can erase entirely.

### Plain resize — `resize`
Left out of the default chain (see Letterbox above) but kept in the
registry for cases where the downstream model genuinely expects a fixed
square input without letterbox padding (e.g., a lightweight OCR crop
classifier that's insensitive to aspect ratio).

## How to re-tune these defaults

1. Run `scripts/run_eda.py` and inspect `brightness_histogram.png`,
   `contrast_histogram.png`, `blur_estimation.png`, and
   `sharpness_distribution.png` for the *actual* downloaded datasets.
2. If brightness is bimodal (well-lit daytime cluster + dark cluster),
   consider gamma > 1.15 or adding a conditional gamma based on measured
   `brightness_mean` per image rather than a single global value.
3. If `blur_estimation.png` shows a long low-variance tail, increase
   `sharpen.amount` incrementally (0.6 → 0.8 → 1.0) and re-check
   `reports/figures/before_after/` samples for over-sharpening artifacts
   (visible haloing around characters is the signal to stop increasing).
4. Every change should be made in `configs/preprocessing_config.yaml` —
   no code changes are needed to re-tune parameters or reorder/toggle steps.
