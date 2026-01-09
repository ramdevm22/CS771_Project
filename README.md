# Security Analysis of PUFs  
**Course Project – CS771 (Introduction to Machine Learning)**  
**Instructor:** Prof. Purushottam Kar, IIT Kanpur  
**Duration:** Jan 2025 – April 2025  

---

## Overview

Physical Unclonable Functions (PUFs) are hardware security primitives that exploit inherent manufacturing variations to generate unique and device-specific responses. While arbiter PUFs are simple and efficient, they are known to be vulnerable to machine learning–based modeling attacks. More complex variants such as XOR-based and multi-level PUFs were proposed to address these weaknesses.

This project analyzes the **security of XOR-arbiter and multi-level PUF architectures** against machine learning attacks. We show that, despite increased architectural complexity, these PUFs can still be accurately modeled using carefully designed feature mappings and linear classifiers.

The work combines **mathematical derivations**, **explicit feature construction**, and **experimental evaluation** using real challenge–response pairs.

---

## Objectives

- Study the vulnerability of XOR-based and multi-level arbiter PUFs  
- Derive an explicit high-dimensional feature mapping for ML-PUFs  
- Train linear models to predict PUF responses  
- Recover feasible internal delay parameters from trained models  
- Evaluate prediction accuracy and robustness across configurations  

---

## Methodology

### Feature Mapping
- Binary challenge bits are transformed into signed representations.
- Interaction terms between challenge bits are constructed.
- A **121-dimensional explicit feature map** is derived to linearize ML-PUF behavior.

### Model Training
- Linear classifiers such as **Logistic Regression** and **Linear SVM** are trained.
- Hyperparameters are tuned to balance accuracy and training time.
- The implementation follows course-imposed constraints on libraries and methods.

### Delay Recovery
- The trained linear model is inverted to recover non-negative delay parameters.
- The inversion is formulated as a constrained optimization problem.
- Lasso regression is used to obtain physically meaningful solutions.

---

## Results

- Achieved **92.63% prediction accuracy** on unseen challenge–response pairs  
- Demonstrated that XOR-based and multi-level PUFs remain vulnerable  
- Successfully recovered feasible delay parameters with negligible error  
- Observed efficient training times using linear models  

These results indicate that architectural hardening alone does not fully protect PUFs from learning-based attacks.

---

## Repository Structure

```
.
├── submit.py          # Core implementation (training, mapping, decoding)
├── report.pdf         # Detailed derivations and experimental results
├── README.md          # Project description
```

---

## How to Run

1. Ensure Python 3.x is installed  
2. Install dependencies:
   ```bash
   pip install numpy scipy scikit-learn
   ```
3. Use the provided functions (`my_fit`, `my_map`, `my_decode`) as required by the evaluation script.

> Note: Function names and allowed libraries strictly follow the course guidelines.

---

## Learning Outcomes

- Gained understanding of ML-based attacks on hardware security primitives  
- Learned to derive explicit feature maps from system equations  
- Applied constrained optimization for model inversion  
- Strengthened ability to connect theory with experimental validation  

---

## Acknowledgements

This project was completed as part of **CS771: Introduction to Machine Learning** at **IIT Kanpur**, under the guidance of **Prof. Purushottam Kar**.
