# WP300 Synthetic Data Generation for SD4EO Project code

## Table of Contents

1. [Introduction](#introduction)
2. [Funding](#funding)
3. [Sub-repositories](#sub-repositories)
   - [UC1: Crop Field Categorisation](#uc1-crop-field-categorisation)
   - [UC2: Human Settlement Categorisation](#uc2-human-settlement-categorisation)
   - [UC3: Photovoltaic Panels Monitoring](#uc3-photovoltaic-panels-monitoring)
4. [Contributing](#contributing)
5. [Licenses](#license)

## Funding

This research work has been funded by the European Space Agency (ESA) under the FutureEO program and the SD4EO project (Contract No.: 4000142334/23/I-DT), supervised by the ESA Φ-lab.

## Introduction

This repository houses the sub-repositories developed for the WP300 work package within the SD4EO project, led by ESA. WP300 focuses on synthetic data generation through artificial intelligence techniques, delivering innovative solutions in an open-source environment. Third-party code has been adapted to expedite the development of various proof-of-concept implementations, organized into sub-repositories corresponding to specific use cases (UCs). For more information on the project objectives, please visit the [SD4EO Project Page](https://eo4society.esa.int/projects/sd4eo/).


## Sub-repositories

Each sub-repository corresponds to a unique use case with distinct goals and methodologies for synthetic data generation.

### UC1: Crop Field Categorisation for Resource Monitoring and Management

- **Sub-folder:** `croptexsynthmethod1`
- **Overview:** Contains code for the mcPS algorithm, an extension of the Portilla-Simoncelli procedural texture generation approach. This implementation synthesizes crop field imagery based on spectral data from Sentinel-1 and Sentinel-2 instruments.
- **Additional Details:** Includes an extended version of Alexei Efros's Image Quilting algorithm, adapted for arbitrary-channel imagery and optimized for patch placement. The goal is to generate synthetic textures that can be utilized to monitor and manage crop resources.

- **Sub-folder:** `chaotic-aggregation`
- **Overview:** Implements a novel polygon covering solution to generate reference images of large-scale monoculture fields, which are needed by the mcPS algorithm. This method was developed specifically for the SD4EO project to align with project constraints.

### UC2: Human Settlement Categorisation for Energy Consumption Monitoring and Management

- **Sub-folder:** `uc2conditioneddiffusionmodel`
- **Overview:** Hosts the training and inference code for diffusion models designed to generate urban imagery as seen by Sentinel-2 in the visible and near-infrared bands. Each spectral band employs a unique model to ensure accurate synthetic data generation.
- **Additional Details:** This code builds upon the ControlNet codebase and Stable Diffusion 1.5, adapted for compatibility with the latest `pytorch_lighting` API, enhancing training and inference processes.

### UC3: Photovoltaic Panels Monitoring to Evaluate Self-Consumption Power Generation

- **Sub-folder:** `UC3SolarPanels`
- **Overview:** Contains inference code for a diffusion model that generates 1m/pixel aerial imagery through Hugging Face’s interface. This synthetic imagery helps to detect rooftop candidates and incorporate solar panels for enhanced monitoring of photovoltaic panels.
- **Additional Details:** Image modifications allow for precise rooftop candidate localization and synthetic solar panel integration, contributing to power generation self-consumption analysis.

Each synthetic image generation process also includes label creation to support use case training and validation.


## Contributing

We welcome contributions that enhance model performance, data synthesis techniques, or add new features. Please create a pull request or raise an issue if you have any questions or suggestions.


## Licenses

Each sub-repository is open-sourced under a different licence: mainly MIT License and APACHE licence version 2.0. See the `LICENSE` file withing each sub-folder for more details.

--- 
The WP300 work package aims to deliver high-quality synthetic data to complement real-world datasets across the outlined use cases. Thank you for your interest and contributions to our synthetic data solutions.
```
