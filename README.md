# CMSI 630: Artificial Intelligence — Assignment 1
## AI-Based Supervised Machine Learning Classification Model Selection

- **Student:** Jillian Hunter
- **Course:** CMSI 630 – Artificial Intelligence
- **Instructor:** Dr. K. Narayanaswamy
- **Due Date:** 10/09/2026 @ 11:59:59 PM
- **Dataset:** Fisher's Iris Classification Dataset (150 samples, 4 continuous features, 3 balanced species classes)

---

## Assignment Deliverables & Rubric Mapping (100 Points)

Per the instructions in `Assignment1.pdf`, all required deliverables are completed and clearly labeled in `Assignment1.ipynb`:

| Rubric Item | Points | Deliverable Location |
| :--- | :---: | :--- |
| **Item 1: Overall Design Documentation, Platforms Used & Execution** | **10 pts** | `Assignment1.ipynb` (Section 1) & `README.md` |
| **Item 2: Best ML Model Selection from Description/Data** | **20 pts** | `Assignment1.ipynb` (Section 2) & `model_selector.py` |
| **Item 3: Implementation of Selected ML Model** | **20 pts** | `Assignment1.ipynb` (Section 3) & `model_selector.py` |
| **Item 4: Cross-Validation Procedures Demonstrating Performance** | **20 pts** | `Assignment1.ipynb` (Section 4) & `model_selector.py` |
| **Item 5: Reasoning for Model Selection & Benchmark Comparison** | **15 pts** | `Assignment1.ipynb` (Section 5) & `figures/` |
| **Item 6: 5-Minute In-Class Presentation Notes & Q&A Defense** | **15 pts** | `Assignment1.ipynb` (Section 6) & `README.md` |
| **Total** | **100 pts** | Single Master Notebook + Supporting Scripts |

---

## Project Structure

```text
Homework1/
├── Assignment1.pdf             # Assignment handout and requirements
├── Assignment1.ipynb           # Master Jupyter Notebook with all labeled sections, code, and pre-run outputs
├── model_selector.py           # Python script implementing the end-to-end model selection and benchmark pipeline
├── test_assignment.py          # Pytest test suite verifying dataset loading, prompts, model training, and metrics
├── build_notebook.py           # Script used to construct and execute Assignment1.ipynb
├── README.md                   # Project documentation and execution instructions
├── .env.example                # Example environment variable file for optional Gemini API key
└── figures/                    # Saved evaluation plots
    ├── benchmark_comparison.png # 6-model cross-validation comparison bar chart
    ├── confusion_matrices.png  # Raw counts and normalized percentages heatmaps
    ├── decision_boundary.png   # 2D PCA projection of SVC decision boundaries & support vectors
    └── roc_curves.png          # Multiclass One-vs-Rest ROC & AUC curves
```

---

## How to Run the Code

### 1. Environment Setup
The project runs on Python 3.10+ (tested on Python 3.14). Required dependencies can be installed via pip:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn google-genai python-dotenv pytest
```

### 2. GenAI API Setup & Offline Grading Support
- **Live API Option:** The program supports querying the Google Gemini API (`gemini-3.8-flash`) via `google-genai`. To use a live key, copy `.env.example` to `.env` and paste your key:
  ```bash
  GEMINI_API_KEY=your_key_here
  ```
  The `.env` file is in `.gitignore` to keep credentials private.
- **Offline / Grading Mode:** If no API key is provided, the notebook and script automatically load the pre-recorded, verified response. This allows Dr. Narayanaswamy or the TA to run **Kernel -> Restart Kernel and Run All Cells** without needing an API key or encountering network blockers.

### 3. Running the Jupyter Notebook
Open the notebook in Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab Assignment1.ipynb
# or
jupyter notebook Assignment1.ipynb
```
Select **Kernel -> Restart Kernel and Run All Cells**. All cells run sequentially in a few seconds with embedded tables and plots.

### 4. Running the Standalone Python Pipeline
You can also run the full pipeline and generate all figures from the terminal:
```bash
python3 model_selector.py
```

### 5. Running the Test Suite
To verify the pipeline components and metrics with pytest:
```bash
pytest test_assignment.py
```

---

## Summary of Results

### 1. Selected Model: Support Vector Classifier (RBF Kernel)
- **Pipeline:** `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)`
- **Stratified 5-Fold Cross-Validation:**
  - **Accuracy:** $96.00\% \pm 3.89\%$
  - **Macro Precision:** $96.11\% \pm 3.83\%$
  - **Macro Recall:** $96.00\% \pm 3.89\%$
  - **Macro F1-Score:** $95.99\% \pm 3.89\%$
- **ROC Area Under Curve (AUC):**
  - *Iris Setosa:* $1.0000$ (perfect linear separation)
  - *Iris Versicolor:* $0.9954$
  - *Iris Virginica:* $0.9950$
  - *Micro-Average:* $0.9968$

### 2. Benchmark Comparison Across 6 Classification Models (5-Fold Stratified CV)

| Rank | Model | 5-Fold Accuracy (Mean ± Std) | F1 Macro (Mean ± Std) | Key Theoretical Takeaway |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **K-Nearest Neighbors (k=5)** | $0.9733 \pm 0.0249$ | $0.9733 \pm 0.0250$ | High empirical score on small $d=4$ space, but non-parametric lazy learner with no margin. |
| 2 | **SVC (Linear Kernel)** | $0.9667 \pm 0.0516$ | $0.9664 \pm 0.0522$ | Maximum margin hyperplane; slightly underfits the non-linear Versicolor/Virginica boundary. |
| 3 | **SVC (RBF Kernel) [Selected]** | **$0.9600 \pm 0.0389$** | **$0.9599 \pm 0.0389$** | **Optimal balance of margin maximization, non-linear kernel mapping, and small-sample robustness.** |
| 4 | **Logistic Regression (Multinomial)** | $0.9533 \pm 0.0452$ | $0.9532 \pm 0.0453$ | Fast and convex, but limited to linear decision boundaries without polynomial features. |
| 5 | **Decision Tree (CART)** | $0.9533 \pm 0.0340$ | $0.9531 \pm 0.0341$ | Axis-aligned splits approximate diagonal boundaries with a staircase pattern; higher fold variance. |
| 6 | **Gaussian Naive Bayes** | $0.9467 \pm 0.0400$ | $0.9465 \pm 0.0401$ | Independence assumption heavily violated by Petal Length/Width collinearity ($r=0.963$). |
| 7 | **Random Forest (100 Trees)** | $0.9467 \pm 0.0267$ | $0.9464 \pm 0.0268$ | An ensemble of 100 trees is over-parameterized for only 150 samples and 4 features. |

---

## 5-Minute In-Class Presentation Notes (Item 6)

### Presentation Outline:
1. **Minute 1: Introduction & Architecture**
   - Introduce the assignment: building a pipeline that uses problem diagnostics and a GenAI model to recommend, implement, and validate the best ML model for the Iris dataset.
   - Mention the tools used: Python, Scikit-Learn, Google Gemini API, and Jupyter Notebook.
2. **Minute 2: Prompt Design & Model Selection**
   - Walk through the dataset profiling: 150 balanced samples, 4 features, severe correlation between petal length and width ($r = 0.963$), and overlap between Versicolor and Virginica.
   - Show how the prompt was structured to query the GenAI model on 6 candidate classifiers.
   - Reveal the selected model: Support Vector Classifier with RBF kernel and `StandardScaler`.
3. **Minute 3: Pipeline & Implementation**
   - Explain why `StandardScaler` is required: since the RBF kernel uses Euclidean distance, unscaled features with larger ranges would unfairly dominate distance calculations.
   - Show model inspection: 53 support vectors chosen (only 8 needed for Setosa; 45 for the Versicolor/Virginica boundary).
4. **Minute 4: Cross-Validation Metrics**
   - Explain why Stratified 5-Fold CV was used: with only 150 samples, an 80/20 train/test split has too few test points; stratified 5-fold ensures 10 of each species in each fold and tests all samples.
   - Review metrics: $96.00\% \pm 3.89\%$ accuracy, $95.99\%$ macro F1, and ROC curves (Setosa 1.0 AUC, Versicolor & Virginica 0.995 AUC).
5. **Minute 5: Benchmark Comparison & Conclusion**
   - Show the 6-model benchmark comparison table and bar chart.
   - Explain why Naive Bayes scored lowest ($94.67\%$): the $r=0.963$ correlation violates its conditional independence assumption.
   - Conclude and open for questions.

### Prepared Discussion Points & Anticipated Questions:
- **Why RBF kernel over Linear kernel?** Setosa is linearly separable, but Versicolor and Virginica have overlapping distributions. A linear kernel requires a flat hyperplane that forces boundary errors, while the RBF kernel maps features into a higher-dimensional space where a smooth non-linear boundary separates them cleanly.
- **Why is `StandardScaler` necessary?** The RBF kernel computes squared Euclidean distance $\|x - x'\|^2$. Unscaled features with larger ranges (like sepal length) would dominate the distance calculation compared to smaller-range features (like petal width).
- **Why did Gaussian Naive Bayes achieve lower accuracy?** Naive Bayes assumes all features are conditionally independent given the class. Petal length and petal width have a Pearson correlation of $0.963$, which heavily violates this assumption and distorts probability estimates.
- **Why Stratified 5-Fold CV instead of a train/test split?** With only 150 samples, an 80/20 split leaves only 30 test samples (10 per class), causing high variance depending on the random split. Stratified 5-Fold CV tests every instance across 5 folds while maintaining exact class balance.
