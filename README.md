Speaker Diarization using Pyannote

This repository contains a speaker diarization project built using Pyannote, exploring multiple approaches to segmentation and diarization, including experimental fine-tuning, reference training pipelines, and a deployable Streamlit application.

The project reflects an iterative machine learning workflow, where different strategies were explored, evaluated, and compared based on stability, scalability, and real-world usability.

Project Structure Overview
.
├── FineTune_Segmentation/
├── Segmentation_model_train/
├── Streamlit/
└── README.md


Each folder represents a distinct stage or approach in the project.

1. FineTune_Segmentation

(Experimental – Exploratory Work)

This folder contains my own attempts at fine-tuning the Pyannote segmentation model.

What was done

Initial segmentation fine-tuning experiments were performed locally using VS Code.

As the dataset size increased, memory and runtime issues were encountered.

To address resource limitations, experiments were migrated to Google Colab.

Multiple training and configuration trials were conducted to understand:

segmentation behavior

speaker activity probabilities

scalability challenges

Important Notes

Some notebooks/scripts include exploratory or unused cells, retained intentionally to reflect the experimentation process.

These experiments helped identify practical limitations of segmentation fine-tuning under constrained resources.

This folder should be considered research-oriented, not production-ready.

📌 Purpose: Learning, experimentation, and failure analysis.

2. Segmentation_model_train

Contents

Train/test setup for segmentation model

Supporting scripts and configurations

Streamlit-based utilities related to the training workflow

Important Clarification

This implementation is included for study, comparison, and understanding best practices.

It served as a benchmark to compare against my own fine-tuning attempts.

The primary contributions of this project are not limited to this folder.

📌 Purpose: Reference and learning support.

3. Streamlit

(Primary & Deployable Component)

This folder contains the Streamlit application for speaker diarization.

Key Features

Accepts user-uploaded audio files

Performs audio preprocessing:

format conversion to WAV

mono channel enforcement

resampling to 16 kHz

Feeds standardized audio into the Pyannote diarization pipeline

Outputs speaker-wise segmentation results

Why this approach

Avoids the heavy computational cost of model fine-tuning

More stable across varied audio inputs

Suitable for real-time and production-style usage

📌 Purpose: Practical, scalable, and user-facing diarization system.

Design Rationale

Multiple approaches were explored to understand the trade-offs between accuracy, complexity, and deployability:

Fine-tuning segmentation models can improve performance but requires:

large labeled datasets

significant computational resources

Preprocessing audio and using pretrained pipelines provides:

faster inference

lower cost

easier deployment

Based on these observations, the Streamlit-based preprocessing + pipeline approach was chosen as the primary solution.

Future Improvements

Re-attempt segmentation fine-tuning with:

cleaner datasets

better GPU resources

Quantitative evaluation of diarization accuracy

Enhanced Streamlit UI and result visualization

Support for longer audio files and batch processing
