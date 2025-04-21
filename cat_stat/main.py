import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

# ---------- Helper Functions ----------

def sanitize_filename(text: str) -> str:
    """Cleans text for safe filenames by replacing special characters."""
    return re.sub(r'\W+', '_', text)

def clean_text(text: str) -> str:
    """Helper function to clean text lines for PDF rendering."""
    return text.strip()

def plot_stacked_percentage_bar(df: pd.DataFrame, x_col: str, hue_col: str, title: str, filename: str) -> None:
    """Creates and saves a stacked percentage bar chart."""
    pivot_table = pd.crosstab(df[x_col], df[hue_col])
    pivot_table_percentage = (pivot_table / pivot_table.sum().sum()) * 100
    percentage_labels = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    cmap = plt.get_cmap("Blues")
    colors = [cmap(i / len(pivot_table.columns)) for i in range(len(pivot_table.columns))]

    ax = pivot_table_percentage.plot(
        kind='bar', stacked=True, color=colors, figsize=(9, 5), edgecolor='black'
    )

    for i, bars in enumerate(ax.containers):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height / 2
                try:
                    percent = percentage_labels.iloc[int(round(bar.get_x())), i]
                    ax.text(x, y, f'{percent:.1f}%', ha='center', va='center', fontsize=10, color='black', weight='bold')
                except (IndexError, KeyError):
                    continue

    plt.title(title, fontsize=14)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title=hue_col.replace("_", " ").title(), bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✅ Stacked bar chart saved as '{filename}'.")
    plt.close()

# ---------- Main Analysis Function ----------

def get_analysis(df: pd.DataFrame, col1: str, col2: str) -> None:
    """
    Perform Chi-Square test and Cramér's V between two categorical variables.
    Saves a Markdown report and a PNG chart.
    """
    # Clean column names for filenames
    col1_clean = sanitize_filename(col1)
    col2_clean = sanitize_filename(col2)

    # Define output filenames
    md_filename = f"stat_analysis_{col1_clean}_vs_{col2_clean}.md"
    png_filename = f"stacked_bar_{col1_clean}_vs_{col2_clean}.png"

    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Chi-Square Test
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Cramér's V
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

    # Interpretations
    chi2_conclusion = (
        f"**Conclusion:** `{col1}` and `{col2}` **are statistically related** (p-value < 0.05)."
        if p_val < 0.05 else
        f"**Conclusion:** No significant relationship between `{col1}` and `{col2}` (p-value > 0.05)."
    )

    cramers_conclusion = (
        "🔹 **Conclusion:** The association is **very weak or negligible**."
        if cramers_v < 0.1 else
        "🔹 **Conclusion:** There is a **weak association**."
        if cramers_v < 0.3 else
        "🔹 **Conclusion:** There is a **moderate association**."
        if cramers_v < 0.5 else
        "🔹 **Conclusion:** The association is **strong**."
    )

    # Create markdown report
    report_md = f"""# 📊 Statistical Analysis: `{col1}` vs `{col2}`

## 1️⃣ Chi-Square Test for Independence
- **Chi-Square Statistic**: {chi2_stat:.4f}
- **p-value**: {p_val:.4e}
- **Degrees of Freedom**: {dof}

{chi2_conclusion}

## 2️⃣ Cramér’s V (Strength of Association)
- **Cramér’s V**: {cramers_v:.4f}

{cramers_conclusion}

## 3️⃣ Visualization
![Stacked Bar Chart]({png_filename})
"""

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"✅ Markdown report saved as '{md_filename}'.")

    # Generate and save the plot
    plot_stacked_percentage_bar(df, col1, col2, f"Stacked Percentage Bar Chart: {col1} vs {col2}", png_filename)

# ---------- Example Call ----------

# Example usage (you would replace `df`, `column1`, `column2` appropriately)
# cat_stat_analysis(df, "Gender", "Purchase")
