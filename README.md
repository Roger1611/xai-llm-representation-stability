# Representation and Stability Analysis of Transformer Language Models

This repository contains the experimental code and results for a study on explainable artificial intelligence applied to transformer-based language models. The work focuses on analyzing how semantic representations form and how stable these representations are across model layers and model scales.

The goal of this study is to examine internal model behavior through hidden representations rather than relying on post-hoc explanation methods. All experiments are conducted using pretrained transformer models without modifying their architectures.

---

## Research Focus

The main focus of this work is to study:

- How semantic distinctions are encoded in transformer hidden states  
- How representation stability changes across different layers  
- Whether explanation behavior remains consistent as model size increases  
- Whether observed patterns generalize across transformer architectures  

The experiments are conducted using DistilBERT and BERT-base models.

---

## Experiments

### Experiment 1: Representation Formation
This experiment examines how ambiguous tokens such as the word "bank" are represented in different semantic contexts. Hidden states are extracted from the model and visualized to observe whether distinct semantic clusters emerge.

### Experiment 2: Explanation Stability
This experiment evaluates how consistent internal representations remain when extracted from different layers of the model. The goal is to understand how explanation stability evolves with depth.

### Experiment 3: Layer-wise Stability Analysis
This experiment provides a quantitative analysis of representation variation across layers. It identifies layers where semantic representations become more stable.

### Experiment 4: Scale Validation
This experiment compares representation stability between smaller and larger transformer models. The objective is to verify whether the observed behavior persists as model capacity increases.

---

## Repository Structure

