#!/usr/bin/env python3
"""
ICU Subspecialty Classification and Analysis

Classifies ICU questions into subspecialties and analyzes scores by category.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path('/mnt/data1/klug/datasets/meditron/Meditron_ICU.xlsx')
OUTPUT_DIR = Path('/home/klug/icu_projects/icu_llm_alignment/output')

# Evaluation columns
EVAL_COLS = [
    'Eval Alignment with Guidelines',
    'Eval Question Comprehension',
    'Eval Logical Reasoning',
    'Eval Relevance & Completeness',
    'Eval Harmlessness',
    'Eval Fairness',
    'Eval Contextual Awareness',
    'Eval Your Confidence',
    'Eval Model Confidence',
    'Eval Communication & Clarity'
]

DIMENSION_NAMES = [
    'Alignment with Guidelines',
    'Question Comprehension',
    'Logical Reasoning',
    'Relevance & Completeness',
    'Harmlessness',
    'Fairness',
    'Contextual Awareness',
    'Rater Confidence',
    'Model Confidence',
    'Communication & Clarity'
]

# Subspecialty definitions with keywords
SUBSPECIALTIES = {
    'Cardiovascular': {
        'keywords': [
            'cardiac arrest', 'cardiogenic shock', 'ecmo', 'va-ecmo', 'vv-ecmo',
            'myocardial infarction', 'stemi', 'nstemi', 'heart', 'cardiac',
            'swan-ganz', 'picco', 'hemodynamic', 'inotrope', 'vasopressor',
            'dobutamine', 'norepinephrine', 'arrhythmia', 'fibrillation',
            'tachycardia', 'bradycardia', 'pacemaker', 'cabg', 'valve',
            'tamponade', 'pulmonary embolism', 'aortic', 'cardiac surgery',
            'rosc', 'cpr', 'resuscitation'
        ],
        'exclude': ['septic shock']
    },
    'Respiratory': {
        'keywords': [
            'ards', 'respiratory failure', 'mechanical ventilation', 'ventilator',
            'intubation', 'extubation', 'tracheostomy', 'tracheotomie', 'weaning',
            'oxygen', 'hypoxemia', 'hypoxia', 'pneumothorax', 'asthma', 'copd',
            'bronchospasm', 'hemoptysis', 'prone position', 'peep', 'fio2',
            'tidal volume', 'respiratory', 'lung', 'pulmonary edema'
        ],
        'exclude': ['pulmonary embolism']
    },
    'Sepsis & Infectious Disease': {
        'keywords': [
            'sepsis', 'septic shock', 'infection', 'antibiotic', 'antimicrobial',
            'meningitis', 'pneumonia', 'bacteremia', 'fungal', 'viral',
            'covid', 'influenza', 'malaria', 'abscess', 'cellulitis',
            'endocarditis', 'osteomyelitis', 'necrotizing', 'fever',
            'inflammatory', 'procalcitonin', 'lactate', 'source control',
            'cultures', 'resistant', 'mrsa', 'pseudomonas', 'candida',
            'pneumocystis', 'immunocompromised'
        ],
        'exclude': []
    },
    'Neurological': {
        'keywords': [
            'traumatic brain injury', 'tbi', 'stroke', 'subarachnoid', 'sah',
            'intracranial', 'icp', 'cerebral', 'brain', 'neurological',
            'neuro', 'seizure', 'epilepsy', 'encephalopathy', 'coma',
            'glasgow', 'gcs', 'pupil', 'herniation', 'vasospasm',
            'neuropronostication', 'brain death', 'eeg', 'ct scan brain',
            'mri brain', 'epidural hematoma', 'subdural', 'contusion'
        ],
        'exclude': []
    },
    'Trauma & Burns': {
        'keywords': [
            'trauma', 'burn', 'fracture', 'hemorrhage', 'bleeding',
            'transfusion', 'massive transfusion', 'coagulopathy',
            'polytrauma', 'injury', 'accident', 'crash', 'fall',
            'laceration', 'contusion', 'spleen', 'liver laceration',
            'damage control', 'tourniquet'
        ],
        'exclude': ['subarachnoid hemorrhage', 'intracranial', 'brain injury']
    },
    'Renal': {
        'keywords': [
            'acute kidney injury', 'aki', 'renal failure', 'dialysis',
            'crrt', 'hemodialysis', 'hemofiltration', 'oliguria', 'anuria',
            'uremia', 'creatinine', 'urea', 'electrolyte', 'hyperkalemia',
            'hyponatremia', 'rhabdomyolysis', 'nephrotoxic'
        ],
        'exclude': []
    },
    'Toxicology': {
        'keywords': [
            'overdose', 'intoxication', 'poisoning', 'toxicity', 'toxic',
            'paracetamol', 'acetaminophen', 'drug', 'suicidal attempt',
            'ingestion', 'antidote', 'n-acetylcysteine', 'charcoal',
            'methanol', 'ethylene glycol', 'opioid', 'benzodiazepine'
        ],
        'exclude': []
    },
    'Hepatic & GI': {
        'keywords': [
            'liver failure', 'hepatic', 'cirrhosis', 'encephalopathy hepatic',
            'pancreatitis', 'gi bleeding', 'gastrointestinal', 'varices',
            'ascites', 'hepatorenal', 'meld', 'child pugh', 'nash',
            'transplant liver', 'biliary', 'cholangitis'
        ],
        'exclude': []
    },
    'Metabolic & Endocrine': {
        'keywords': [
            'diabetic ketoacidosis', 'dka', 'hypoglycemia', 'hyperglycemia',
            'thyroid', 'adrenal', 'cortisol', 'insulin', 'glucose',
            'acid-base', 'acidosis', 'alkalosis', 'electrolyte',
            'hypercalcemia', 'hypocalcemia', 'hypomagnesemia'
        ],
        'exclude': ['lactic acidosis', 'septic']
    },
    'Ethics & End-of-Life': {
        'keywords': [
            'withdrawal', 'end of life', 'palliative', 'comfort care',
            'organ donation', 'brain death', 'prognosis', 'futility',
            'family meeting', 'goals of care', 'do not resuscitate', 'dnr',
            'advance directive', 'surrogate', 'ethics'
        ],
        'exclude': []
    },
    'Procedures & Monitoring': {
        'keywords': [
            'central line', 'central venous', 'arterial line', 'catheter',
            'lumbar puncture', 'thoracentesis', 'paracentesis', 'bronchoscopy',
            'ultrasound', 'echocardiography', 'monitoring', 'supervision',
            'training', 'procedure', 'technique', 'insertion'
        ],
        'exclude': []
    }
}


def classify_question(question_text):
    """
    Classify a question into an ICU subspecialty based on keywords.
    Returns the primary subspecialty and confidence score.
    """
    question_lower = question_text.lower()

    scores = {}

    for subspecialty, config in SUBSPECIALTIES.items():
        score = 0

        # Check for keywords
        for keyword in config['keywords']:
            if keyword in question_lower:
                # Weight longer phrases more heavily
                weight = len(keyword.split())
                score += weight

        # Check for exclusions
        for exclude in config.get('exclude', []):
            if exclude in question_lower:
                score = max(0, score - 2)

        scores[subspecialty] = score

    # Get the subspecialty with highest score
    if max(scores.values()) == 0:
        return 'General ICU', 0

    best_subspecialty = max(scores, key=scores.get)
    confidence = scores[best_subspecialty]

    return best_subspecialty, confidence


def classify_all_questions(df):
    """Classify all unique questions and return a mapping."""

    questions = df['Question'].unique()
    classifications = {}

    for q in questions:
        subspecialty, confidence = classify_question(q)
        classifications[q] = {
            'subspecialty': subspecialty,
            'confidence': confidence
        }

    return classifications


def add_subspecialty_to_df(df, classifications):
    """Add subspecialty column to dataframe."""
    df['Subspecialty'] = df['Question'].map(
        lambda q: classifications[q]['subspecialty']
    )
    df['Classification_Confidence'] = df['Question'].map(
        lambda q: classifications[q]['confidence']
    )
    return df


def analyze_by_subspecialty(df):
    """Calculate metrics by subspecialty."""

    results = []

    for subspecialty in df['Subspecialty'].unique():
        subdf = df[df['Subspecialty'] == subspecialty]

        # Overall metrics
        n_questions = subdf['Question'].nunique()
        n_ratings = len(subdf)

        # Calculate mean for each dimension
        dim_means = {}
        for col, name in zip(EVAL_COLS, DIMENSION_NAMES):
            dim_means[name] = subdf[col].mean()

        # Overall mean across all dimensions
        overall_mean = subdf[EVAL_COLS].mean().mean()
        overall_std = subdf[EVAL_COLS].values.flatten().std()

        results.append({
            'Subspecialty': subspecialty,
            'N_Questions': n_questions,
            'N_Ratings': n_ratings,
            'Overall_Mean': overall_mean,
            'Overall_SD': overall_std,
            **dim_means
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('N_Questions', ascending=False)

    return results_df


def create_subspecialty_summary_table(df):
    """Create a summary table for subspecialties."""

    summary = df.groupby('Subspecialty').agg({
        'Question': 'nunique',
        **{col: ['mean', 'std'] for col in EVAL_COLS}
    }).round(2)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                       for col in summary.columns]
    summary = summary.rename(columns={'Question_nunique': 'N_Questions'})

    # Calculate overall mean per subspecialty
    overall_means = df.groupby('Subspecialty')[EVAL_COLS].mean().mean(axis=1)
    summary['Overall_Mean'] = overall_means.round(2)

    summary = summary.sort_values('N_Questions', ascending=False)

    return summary


def plot_subspecialty_scores(df):
    """Create bar plot of overall scores by subspecialty."""

    # Calculate mean scores by subspecialty
    subspecialty_scores = df.groupby('Subspecialty')[EVAL_COLS].mean().mean(axis=1)
    subspecialty_scores = subspecialty_scores.sort_values(ascending=True)

    # Calculate SEM for error bars
    subspecialty_sem = df.groupby('Subspecialty')[EVAL_COLS].apply(
        lambda x: x.values.flatten().std() / np.sqrt(len(x))
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(subspecialty_scores)))

    bars = ax.barh(range(len(subspecialty_scores)), subspecialty_scores.values,
                   color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(subspecialty_scores)))
    ax.set_yticklabels(subspecialty_scores.index)
    ax.set_xlabel('Mean Score (1-5)', fontsize=12)
    ax.set_title('Overall Evaluation Scores by ICU Subspecialty',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(1, 5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, subspecialty_scores.values)):
        ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=10)

    # Add count labels
    question_counts = df.groupby('Subspecialty')['Question'].nunique()
    for i, subspecialty in enumerate(subspecialty_scores.index):
        n_q = question_counts[subspecialty]
        ax.text(1.1, i, f'(n={n_q})', va='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subspecialty_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'subspecialty_scores.pdf', bbox_inches='tight')
    plt.close()


def plot_subspecialty_heatmap(df):
    """Create heatmap of dimension scores by subspecialty."""

    # Calculate mean scores
    heatmap_data = df.groupby('Subspecialty')[EVAL_COLS].mean()
    heatmap_data.columns = DIMENSION_NAMES

    # Sort by number of questions
    question_counts = df.groupby('Subspecialty')['Question'].nunique()
    heatmap_data = heatmap_data.loc[question_counts.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                center=3, vmin=1, vmax=5, ax=ax,
                cbar_kws={'label': 'Mean Score'})

    ax.set_title('Evaluation Scores by ICU Subspecialty and Dimension',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Dimension', fontsize=12)
    ax.set_ylabel('ICU Subspecialty', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subspecialty_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'subspecialty_heatmap.pdf', bbox_inches='tight')
    plt.close()


def plot_subspecialty_distribution(df):
    """Create pie chart of question distribution by subspecialty."""

    question_counts = df.groupby('Subspecialty')['Question'].nunique()
    question_counts = question_counts.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.Set3(np.linspace(0, 1, len(question_counts)))

    wedges, texts, autotexts = ax.pie(
        question_counts.values,
        labels=question_counts.index,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(question_counts.values))})',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )

    plt.setp(autotexts, size=9)
    plt.setp(texts, size=10)

    ax.set_title('Distribution of Questions by ICU Subspecialty',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subspecialty_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'subspecialty_distribution.pdf', bbox_inches='tight')
    plt.close()


def plot_subspecialty_boxplot(df):
    """Create boxplot of scores by subspecialty."""

    # Calculate overall mean per rating
    df_plot = df.copy()
    df_plot['Overall_Score'] = df_plot[EVAL_COLS].mean(axis=1)

    # Order by median
    order = df_plot.groupby('Subspecialty')['Overall_Score'].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(data=df_plot, x='Subspecialty', y='Overall_Score',
                order=order, palette='RdYlGn', ax=ax)

    ax.set_xlabel('ICU Subspecialty', fontsize=12)
    ax.set_ylabel('Overall Score (Mean Across Dimensions)', fontsize=12)
    ax.set_title('Distribution of Overall Scores by ICU Subspecialty',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subspecialty_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'subspecialty_boxplot.pdf', bbox_inches='tight')
    plt.close()


def statistical_comparison(df):
    """Perform statistical comparison between subspecialties."""

    # Calculate overall mean per rating
    df['Overall_Score'] = df[EVAL_COLS].mean(axis=1)

    # Kruskal-Wallis test (non-parametric ANOVA)
    groups = [group['Overall_Score'].values for name, group in df.groupby('Subspecialty')]
    h_stat, p_value = stats.kruskal(*groups)

    results = {
        'Test': 'Kruskal-Wallis',
        'H-statistic': h_stat,
        'p-value': p_value,
        'Significant': p_value < 0.05
    }

    return results


def print_classification_examples(df, classifications, n=5):
    """Print example classifications for verification."""

    print("\n=== Sample Classifications ===\n")

    for subspecialty in df['Subspecialty'].unique():
        questions = df[df['Subspecialty'] == subspecialty]['Question'].unique()[:n]
        print(f"\n{subspecialty} ({len(df[df['Subspecialty'] == subspecialty]['Question'].unique())} questions):")
        print("-" * 60)
        for q in questions[:2]:
            q_short = ' '.join(q.split())[:150]
            conf = classifications[q]['confidence']
            print(f"  [{conf}] {q_short}...")


def main():
    """Main subspecialty analysis pipeline."""

    print("Loading data...")
    df = pd.read_excel(DATA_PATH)

    # Clean rater names
    name_mapping = {
        'RaphaÃ«l Burger': 'Raphaël Burger',
        'AurÃ©lie Leuenberger': 'Aurélie Leuenberger'
    }
    df['Name'] = df['Name'].replace(name_mapping)

    print(f"Loaded {len(df)} ratings for {df['Question'].nunique()} questions")

    # Classify questions
    print("\nClassifying questions into subspecialties...")
    classifications = classify_all_questions(df)
    df = add_subspecialty_to_df(df, classifications)

    # Print classification distribution
    print("\n=== Subspecialty Distribution ===")
    subspecialty_counts = df.groupby('Subspecialty')['Question'].nunique().sort_values(ascending=False)
    for subspecialty, count in subspecialty_counts.items():
        pct = count / df['Question'].nunique() * 100
        print(f"  {subspecialty}: {count} questions ({pct:.1f}%)")

    # Print sample classifications
    print_classification_examples(df, classifications)

    # Analyze by subspecialty
    print("\n\nCalculating subspecialty metrics...")
    subspecialty_analysis = analyze_by_subspecialty(df)

    # Create summary table
    print("\n=== Subspecialty Analysis Summary ===\n")
    display_cols = ['Subspecialty', 'N_Questions', 'N_Ratings', 'Overall_Mean', 'Overall_SD']
    print(subspecialty_analysis[display_cols].to_string(index=False))

    # Save detailed results
    subspecialty_analysis.to_csv(OUTPUT_DIR / 'subspecialty_analysis.csv', index=False)

    # Create simplified table
    simple_table = subspecialty_analysis[['Subspecialty', 'N_Questions', 'N_Ratings',
                                           'Overall_Mean', 'Overall_SD']].copy()
    simple_table['Overall_Mean'] = simple_table['Overall_Mean'].round(2)
    simple_table['Overall_SD'] = simple_table['Overall_SD'].round(2)
    simple_table.to_csv(OUTPUT_DIR / 'subspecialty_summary.csv', index=False)

    # Statistical comparison
    print("\n=== Statistical Comparison ===")
    stat_results = statistical_comparison(df.copy())
    print(f"Kruskal-Wallis H = {stat_results['H-statistic']:.2f}, p = {stat_results['p-value']:.4f}")
    if stat_results['Significant']:
        print("Significant difference between subspecialties (p < 0.05)")

    # Generate plots
    print("\nGenerating subspecialty plots...")

    plot_subspecialty_scores(df)
    print("  - Subspecialty scores bar plot")

    plot_subspecialty_heatmap(df)
    print("  - Subspecialty heatmap")

    plot_subspecialty_distribution(df)
    print("  - Subspecialty distribution pie chart")

    plot_subspecialty_boxplot(df)
    print("  - Subspecialty boxplot")

    # Save classification mapping
    classification_df = pd.DataFrame([
        {'Question': q[:200], 'Subspecialty': v['subspecialty'], 'Confidence': v['confidence']}
        for q, v in classifications.items()
    ])
    classification_df.to_csv(OUTPUT_DIR / 'question_classifications.csv', index=False)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

    return df, subspecialty_analysis, classifications


if __name__ == '__main__':
    df, subspecialty_analysis, classifications = main()
