# MDWD - Maltese Domestic Waste Dataset > MDWD-RAW
https://universe.roboflow.com/um-vbsez/mdwd-maltese-domestic-waste-dataset

Provided by a Roboflow user
License: BY-NC-SA 4.0

# MDWD — Maltese Domestic Waste Dataset

## Project Context

The MDWD forms part of the **Application of AI and Computer Vision to Optimise Cleansing Operations in Malta (AICOM)** project, funded by the Government of Malta Cleansing and Maintenance Division (CMD). The project explores how Artificial Intelligence and Computer Vision can enable more scalable, consistent, and data-driven approaches to municipal cleansing operations and waste monitoring across Malta.

## Overview

The **Maltese Domestic Waste Dataset (MDWD)** is a street-level object-detection dataset designed to support research into automated municipal solid-waste monitoring.

MDWD contains **3,697 high-resolution images** and **11,461 manually annotated waste instances** collected across Malta and Gozo. The images represent realistic urban collection environments and include substantial variation in:

- Location and surrounding urban context
- Illumination and weather conditions
- Object size and viewing distance
- Occlusion and visual clutter
- Camera orientation and capture device
- Waste presentation and placement patterns

Unlike datasets centred on isolated litter items or image-level classification, MDWD provides **instance-level bounding-box annotations** for multiple domestic waste streams encountered within an operational municipal collection system.

## Dataset Classes

The dataset contains five object classes:

### Mixed Waste

Residual household refuse, typically placed in black waste bags.

### Organic Waste

Biodegradable household waste, generally presented in small white or translucent bags under Malta's organic-waste collection scheme.

### Recyclable Material

Recyclable paper, cardboard, plastic and metal packaging, commonly presented in grey or green bags.

### Orange CMD

Distinctive orange waste bags associated with Cleaning and Maintenance Division operations.

### Other Waste

Less frequently encountered or visually heterogeneous waste, including bulky refuse, cardboard boxes, glass bottles and other items collected separately from the principal household waste streams.

## Data Collection and Privacy

Images were collected across Malta and Gozo between winter 2024 and late summer 2025 using a range of consumer-grade mobile devices. Photographs were captured primarily on foot, with a smaller proportion collected from moving vehicles.

Each image represents a distinct waste pile or collection event. Burst photographs and video-derived frames were excluded to reduce near-duplicate imagery.

Visible personally identifiable information falling within the scope of the General Data Protection Regulation was manually redacted before publication.

## Dataset Split

The dataset is divided into:

| Partition | Images | Percentage |
|---|---:|---:|
| Training | 2,958 | 80% |
| Validation | 370 | 10% |
| Test | 369 | 10% |
| **Total** | **3,697** | **100%** |

The original images have a median resolution of approximately **3024 × 4032 pixels**, with an average of **3.1 annotated instances per image**.

## Intended Uses

MDWD is intended for research and development involving:

- Municipal solid-waste detection
- Multi-class object detection
- Street-level environmental monitoring
- Urban cleanliness assessment
- Waste-collection compliance monitoring
- Long-tailed and imbalanced object recognition
- Real-time and edge-deployable detection systems
- Comparison of convolutional and transformer-based detectors
- Vision-language and prompt-based localisation research

The dataset may also support the development of decision-support tools for municipal authorities, cleansing operators and environmental monitoring organisations.

## Benchmark Results

Baseline experiments were conducted using multiple generations of the YOLO detector family and the transformer-based RF-DETR architecture.

The strongest evaluated model, **RF-DETR-M**, achieved:

- **94.49% mAP@50**
- **78.21% mAP@50:95**
- **96.63% precision**
- **90.69% recall**
- **93.56% F1-score**

Smaller detector variants also achieved competitive results, demonstrating that MDWD can support both high-capacity models and compact models intended for real-time deployment.

These results are provided as reproducible reference baselines rather than as an indication that any single architecture is universally optimal.

## Contact

For inquiries, collaboration opportunities, or feedback, please contact [Andrea Filiberto Lucas](mailto:contact@aflucas.com).

**GitHub:** [AFLucas-UOM](https://github.com/AFLucas-UOM)