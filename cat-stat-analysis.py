import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from fpdf import FPDF
import re

def sanitize_filename(text):
    """
    Cleans column names for use in filenames.
    - Removes special characters.
    - Replaces spaces with underscores.
    """
    return re.sub(r'\W+', '_', text)

def statistical_analysis(df, col1, col2):
    """
    Perform Chi-Square test and Cramér's V for two categorical variables.
    Outputs:
    - Markdown report (.md)
    - Stacked percentage bar chart (.png)
    - PDF report with results and visualization (.pdf)
    
    Parameters:
    - df: DataFrame containing the data
    - col1, col2: Two categorical columns to analyze
    """

    # Clean column names for filenames
    col1_clean = sanitize_filename(col1)
    col2_clean = sanitize_filename(col2)

    # File names based on column names
    md_filename = f"stat_analysis_{col1_clean}_vs_{col2_clean}.md"
    png_filename = f"stacked_bar_{col1_clean}_vs_{col2_clean}.png"
    pdf_filename = f"stat_analysis_{col1_clean}_vs_{col2_clean}.pdf"

    # Create a contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Chi-Square Test
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Chi-Square Conclusion
    chi2_conclusion = f"🔹 **Conclusion:** `{col1}` and `{col2}` **are statistically related** (p-value < 0.05)." \
        if p_val < 0.05 else f"🔹 **Conclusion:** No significant relationship between `{col1}` and `{col2}` (p-value > 0.05)."

    # Cramér’s V Calculation
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

    # Cramér’s V Conclusion
    cramers_conclusion = (
        "🔹 **Conclusion:** The association is **very weak or negligible**."
        if cramers_v < 0.1 else
        "🔹 **Conclusion:** There is a **weak association**."
        if cramers_v < 0.3 else
        "🔹 **Conclusion:** There is a **moderate association**."
        if cramers_v < 0.5 else
        "🔹 **Conclusion:** The association is **strong**."
    )

    # Save Markdown Report
    report_md = f"""
    # 📊 Statistical Analysis of `{col1}` vs `{col2}`

    ## 1️⃣ Chi-Square Test for Independence
    - **Chi-Square Statistic**: {chi2_stat:.4f}
    - **p-value**: {p_val:.4e}
    - **Degrees of Freedom**: {dof}

    {chi2_conclusion}

    ## 2️⃣ Cramér’s V (Strength of Association)
    - **Cramér’s V**: {cramers_v:.4f}

    {cramers_conclusion}

    ## 3️⃣ Stacked Percentage Bar Chart
    ![Stacked Bar Chart]({png_filename})
    """

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"✅ Statistical analysis report saved as '{md_filename}'.")

    # Generate and save the plot
    plot_stacked_percentage_bar(df, col1, col2, f"Stacked Percentage Bar Chart: {col1} vs {col2}", png_filename)

    # Generate PDF Report
    generate_pdf_report(md_filename, png_filename, pdf_filename)

def plot_stacked_percentage_bar(df, x_col, hue_col, title, filename):
    """
    Creates a stacked percentage bar chart and saves it dynamically.
    """
    pivot_table = pd.crosstab(df[x_col], df[hue_col])
    pivot_table_percentage = (pivot_table / pivot_table.sum().sum()) * 100
    percentage_labels = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    cmap = plt.get_cmap("Blues")
    colors = [cmap(i / len(pivot_table.columns)) for i in range(len(pivot_table.columns))]

    ax = pivot_table_percentage.plot(kind='bar', stacked=True, color=colors, figsize=(9, 5), edgecolor='black')

    for i, bars in enumerate(ax.containers):  
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height / 2
                percent = percentage_labels.iloc[int(bar.get_x()), i]
                ax.text(x, y, f'{percent:.1f}%', ha='center', va='center', fontsize=10, color='black', weight='bold')

    plt.title(title, fontsize=14)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title=hue_col.replace("_", " ").title(), bbox_to_anchor=(1, 1))

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✅ Stacked bar chart saved as '{filename}'.")
    plt.show()

def generate_pdf_report(md_file, image_file, pdf_filename):
    """
    Converts a markdown report and an image into a PDF with dynamic naming.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    with open(md_file, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = clean_text(line)
            pdf.cell(200, 8, txt=clean_line, ln=True, align="L")

    pdf.ln(10)
    pdf.image(image_file, x=20, w=170)

    pdf.output(pdf_filename, "F")
    print(f"✅ PDF report saved as '{pdf_filename}'.")
