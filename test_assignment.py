"""
Unit tests for Assignment 1: AI-based Supervised ML Model Selection
Verifies that all pipeline components, metrics, and models satisfy course requirements.
"""

import pytest
import numpy as np
import pandas as pd
from model_selector import (
    load_and_profile_iris,
    get_dataset_summary_prompt_context,
    build_user_prompt,
    LLMModelSelector,
    evaluate_selected_model,
    run_multi_model_benchmark,
    SYSTEM_PROMPT,
)


def test_dataset_loading():
    df, X, y, feature_names, target_names = load_and_profile_iris()
    assert len(df) == 150
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(feature_names) == 4
    assert len(target_names) == 3
    assert set(target_names) == {"setosa", "versicolor", "virginica"}


def test_prompt_generation():
    df, X, y, feature_names, target_names = load_and_profile_iris()
    context = get_dataset_summary_prompt_context(df, feature_names, target_names)
    assert "Fisher's Iris Dataset" in context
    assert "Number of Samples (N): 150" in context

    user_prompt = build_user_prompt("Test Problem", context)
    assert "Test Problem" in user_prompt
    assert "Support Vector Classifier" in user_prompt
    assert "Candidate Algorithm Families" in user_prompt


def test_llm_selector_and_code_extraction():
    # Verify extraction and parser functionality with offline verified response
    selector = LLMModelSelector(gemini_api_key="", anthropic_api_key="")
    response = selector.query(SYSTEM_PROMPT, "Test Prompt")
    assert "Selected Model" in response
    assert "Support Vector Classifier" in response

    code = selector.extract_python_code(response)
    assert len(code) > 0
    assert "SVC" in code
    assert "Pipeline" in code


def test_selected_model_evaluation():
    _, X, y, _, target_names = load_and_profile_iris()
    results = evaluate_selected_model(X, y, target_names, n_splits=5)

    summary = results["metrics_summary"]
    assert 0.90 <= summary["Accuracy Mean"] <= 1.0
    assert 0.90 <= summary["Precision Macro Mean"] <= 1.0
    assert 0.90 <= summary["Recall Macro Mean"] <= 1.0
    assert 0.90 <= summary["F1 Macro Mean"] <= 1.0

    cm = results["confusion_matrix"]
    assert cm.shape == (3, 3)
    assert cm.sum() == 150

    # Class 0 (Setosa) must be perfectly classified (50/50)
    assert cm[0, 0] == 50

    # ROC AUC checks
    roc = results["roc_data"]
    assert roc["setosa"]["auc"] == 1.0
    assert roc["versicolor"]["auc"] >= 0.95
    assert roc["virginica"]["auc"] >= 0.95
    assert roc["micro"]["auc"] >= 0.98


def test_multi_model_benchmark():
    _, X, y, _, _ = load_and_profile_iris()
    benchmark_df = run_multi_model_benchmark(X, y, n_splits=5)

    assert len(benchmark_df) == 7
    assert "SVC (RBF Kernel) [AI Selected]" in benchmark_df["Model"].values
    assert "Logistic Regression (Multinomial)" in benchmark_df["Model"].values
    assert "Random Forest (100 Trees)" in benchmark_df["Model"].values

    # Check that all models achieve reasonable performance on Iris (> 90%)
    for acc in benchmark_df["Accuracy Mean"]:
        assert acc >= 0.90
