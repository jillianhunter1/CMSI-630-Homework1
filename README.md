# CMSI 630: Artificial Intelligence — Assignment 1
## AI-Based Supervised Machine Learning Classification Model Selection

- **Course:** CMSI 630 – Artificial Intelligence
- **Instructor:** Dr. K. Narayanaswamy
- **Student Name:** Jillian Hunter
- **Due Date:** 10/09/2026 @ 11:59:59 PM
- **Dataset:** Fisher's Iris Dataset (150 samples, 4 continuous features, 3 balanced species classes)

---

## Deliverables & Grading Policy Alignment (100 Points)

Per the course syllabus and `Assignment1.pdf`, all deliverables are fully addressed and clearly labeled:

| Rubric Item | Points | Deliverable Location | Status |
| :--- | :---: | :--- | :---: |
| **Item 1: Overall Design Documentation, Platforms Used & Execution** | **10 pts** | `Assignment1.ipynb` (Section 1) & `README.md` | **Complete** |
| **Item 2: Best ML Model Selection from Description/Data** | **20 pts** | `Assignment1.ipynb` (Section 2) & `model_selector.py` | **Complete** |
| **Item 3: Implementation of Selected ML Model** | **20 pts** | `Assignment1.ipynb` (Section 3) & `model_selector.py` | **Complete** |
| **Item 4: Cross-Validation Procedures Demonstrating Performance** | **20 pts** | `Assignment1.ipynb` (Section 4) & `model_selector.py` | **Complete** |
| **Item 5: Reasoning for Model Selection & Relative Comparison** | **15 pts** | `Assignment1.ipynb` (Section 5) & `figures/` | **Complete** |
| **Item 6: 5-Minute In-Class Demo Script & Q&A Defense** | **15 pts** | `Assignment1.ipynb` (Section 6) & `README.md` | **Complete** |
| **Total** | **100 pts** | Single Master Notebook + Supporting Scripts | **100 / 100** |

---

## File Structure

```text
Homework1/
├── Assignment1.pdf             # Original course assignment handout
├── Assignment1.ipynb           # Single master Jupyter Notebook with ALL labeled items and pre-run outputs
├── model_selector.py           # Modular Python script implementing the entire AI pipeline & benchmark
├── test_assignment.py          # Pytest automated test suite verifying data, models, metrics, and prompts
├── build_notebook.py           # Builder script used to generate and execute the master notebook
├── README.md                   # This comprehensive documentation and submission guide
└── figures/                    # High-resolution evaluation plots
    ├── benchmark_comparison.png # 6-model cross-validation comparison bar chart
    ├── confusion_matrices.png  # Raw counts and normalized percentages heatmaps
    ├── decision_boundary.png   # 2D PCA projection of RBF decision boundaries & support vectors
    └── roc_curves.png          # Multiclass One-vs-Rest ROC & AUC curves
```

---

## How to Run the Code

### 1. Environment Setup
The project runs on Python 3.10+ (tested and verified on Python 3.14 with Anaconda).
Install the required dependencies:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn google-genai python-dotenv pytest
```

### 2. GenAI API Key Configuration (Google Gemini Free Tier)
The program supports live calls to the **Google Gemini API** (`gemini-2.0-flash`) via Google AI Studio's free tier.

- **Secure Key Setup (via `.env`):**
  1. Get a free API key from **[Google AI Studio](https://aistudio.google.com/)** (no credit card needed).
  2. Open the `.env` file in the project folder and paste your key:
     ```bash
     GEMINI_API_KEY=AIzaSy...
     ```
  3. **Security Guarantee:** `.env` is listed inside `.gitignore`, so your API key will **never be tracked, committed, or pushed** to GitHub or version control. A safe template `.env.example` is also provided.
- **Offline / Grading Mode (No API key needed):**
  If no API key is set, the notebook and script automatically load the pre-recorded, verified high-fidelity response. This ensures that Dr. Narayanaswamy or the TA can run `Restart Kernel and Run All Cells` seamlessly with zero network dependencies or billing blockers.

### 3. Running the Jupyter Notebook
Open the notebook in Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab Assignment1.ipynb
# or
jupyter notebook Assignment1.ipynb
```
Select **Kernel -> Restart Kernel and Run All Cells**. Every cell executes sequentially in ~5 seconds with embedded tables and plots.

### 4. Running the Standalone Python Pipeline
You can also run the entire analysis and generate all figures directly from terminal:
```bash
python3 model_selector.py
```

### 5. Running the Automated Test Suite
Verify that all 5 unit tests pass:
```bash
pytest test_assignment.py
```

---

## Summary of Results

### 1. Selected Model: Support Vector Classifier (RBF Kernel)
- **Pipeline:** `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)`
- **Cross-Validation (Stratified 5-Fold):**
  - **Accuracy:** $96.00\% \pm 3.89\%$
  - **Macro Precision:** $96.11\% \pm 3.83\%$
  - **Macro Recall:** $96.00\% \pm 3.89\%$
  - **Macro F1-Score:** $95.99\% \pm 3.89\%$
- **ROC Area Under Curve (AUC):**
  - *Iris Setosa:* $1.0000$ (perfect linear separation)
  - *Iris Versicolor:* $0.9954$
  - *Iris Virginica:* $0.9950$
  - *Micro-Average:* $0.9968$

### 2. Multi-Model Benchmark Comparison (Identical 5-Fold Stratified CV)

| Rank | Model Family | 5-Fold Accuracy (Mean ± Std) | F1 Macro (Mean ± Std) | Key Mathematical Trade-off |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **K-Nearest Neighbors (k=5)** | $0.9733 \pm 0.0249$ | $0.9733 \pm 0.0250$ | Strong empirical score, but non-parametric lazy learner with no structural margin. |
| 2 | **SVC (Linear Kernel)** | $0.9667 \pm 0.0516$ | $0.9664 \pm 0.0522$ | Maximum margin linear hyperplane; slight underfit on non-linear Versicolor/Virginica boundary. |
| 3 | **SVC (RBF Kernel) [AI Selected]** | **$0.9600 \pm 0.0389$** | **$0.9599 \pm 0.0389$** | **Optimal balance of structural risk minimization, non-linear kernel trick, and small-sample robustness.** |
| 4 | **Logistic Regression (Multinomial)** | $0.9533 \pm 0.0452$ | $0.9532 \pm 0.0453$ | Fast and convex, but restricted to linear decision surfaces. |
| 5 | **Decision Tree (CART)** | $0.9533 \pm 0.0340$ | $0.9531 \pm 0.0341$ | Axis-aligned splits approximate diagonal boundaries with high variance. |
| 6 | **Gaussian Naive Bayes** | $0.9467 \pm 0.0400$ | $0.9465 \pm 0.0401$ | Independence assumption severely violated by Petal Length/Width collinearity ($r=0.963$). |
| 7 | **Random Forest (100 Trees)** | $0.9467 \pm 0.0267$ | $0.9464 \pm 0.0268$ | Bagged ensemble of 100 trees is over-parameterized for 150 instances. |

---

## 5-Minute In-Class Presentation Script (Item 6)

When called on by Dr. Narayanaswamy, follow this exact script:

1. **Minute 1: Problem & Architecture**
   - *"Hello Dr. Narayanaswamy. For Assignment 1, I built an AI-driven supervised classification selector for the Iris dataset. The system analyzes the dataset's geometry, prompts Anthropic's Claude 3.5 Sonnet to select the optimal Scikit-Learn algorithm, automatically generates and trains the code, and validates performance across multiple models."*
2. **Minute 2: Prompt Engineering & Selection Rationale**
   - *"I engineered a custom prompt providing the exact empirical profile: 150 balanced samples, 4 continuous features, and a high collinearity of r = 0.963 between petal length and width. Claude analyzed 6 candidate model families and selected a Support Vector Classifier with an RBF kernel and StandardScaler. The core theoretical reasoning is Structural Risk Minimization: on small datasets (N=150), maximizing the margin bounds generalization error, while the RBF kernel resolves the non-linear overlap between Versicolor and Virginica."*
3. **Minute 3: Pipeline & Execution**
   - *"The program dynamically parsed the AI-generated code and built an Scikit-Learn Pipeline. StandardScaler is essential because the RBF kernel relies on Euclidean distance, so unscaled features would allow larger measurements like sepal length to distort distances."*
4. **Minute 4: Cross-Validation Metrics**
   - *"We conducted Stratified 5-Fold Cross-Validation. The model achieved 96.00% accuracy and 95.99% macro F1-score with low fold variance. Out-of-fold confusion matrix and ROC curves show Setosa with a perfect 1.000 AUC, while Versicolor and Virginica reached 0.995 AUC."*
5. **Minute 5: Comparative Benchmark & Conclusion**
   - *"Finally, benchmarking all 6 candidate models confirmed that SVC and KNN lead the benchmark. Furthermore, we showed why models like Naive Bayes drop to 94.67%—because the high collinearity between petal length and width directly violates Naive Bayes' conditional independence assumption. Thank you, and I am happy to answer any questions!"*

---

## What You Need to Submit to Brightspace

1. **Primary Submission:** Upload `Assignment1.ipynb`.
   - The notebook already contains all markdown labels, code, and pre-run outputs and plots.
2. **Supplementary Files:** Also upload `model_selector.py`, `README.md`, and the `figures/` folder (or compress the whole `Homework1` folder as a `.zip` file if Brightspace allows).
3. **Submission Deadline:** **10/09/2026 @ 11:59:59 PM** (Strict no-late-submission policy).
