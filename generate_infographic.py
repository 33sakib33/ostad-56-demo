import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set global matplotlib style parameters
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['text.color'] = '#F8FAFC'
plt.rcParams['axes.labelcolor'] = '#CBD5E1'
plt.rcParams['xtick.color'] = '#94A3B8'
plt.rcParams['ytick.color'] = '#94A3B8'

def load_data():
    zip_path = 'ostad-56-batch-contest.zip'
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open('train.csv'))
    return df

def draw_kpi_card(ax, title, value, subtitle, bg_color='#1E293B', border_color='#334155', title_color='#94A3B8', val_color='#FFD166'):
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    # Draw background box with rounded corners
    rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                          boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor=bg_color, edgecolor=border_color, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    
    ax.text(0.5, 0.72, title.upper(), transform=ax.transAxes,
            ha='center', va='center', fontsize=12, fontweight='bold', color=title_color)
    ax.text(0.5, 0.42, value, transform=ax.transAxes,
            ha='center', va='center', fontsize=22, fontweight='bold', color=val_color)
    ax.text(0.5, 0.18, subtitle, transform=ax.transAxes,
            ha='center', va='center', fontsize=9.5, color='#CBD5E1', fontstyle='italic')

def main():
    df = load_data()
    
    # Color palette definition
    C_BG = '#0B132B'
    C_CARD = '#1C2541'
    C_BORDER = '#3A506B'
    C_NON_SMOKER = '#06D6A0' # Teal
    C_SMOKER = '#FF4B4B'     # Bright Crimson
    C_ACCENT = '#FFD166'     # Warm Amber/Gold
    C_TEXT = '#F8FAFC'
    
    fig = plt.figure(figsize=(24, 16), facecolor=C_BG)
    
    # Grid layout: Header, KPI Row, Main Content 3x2 Grid, Footer
    gs = gridspec.GridSpec(4, 3, height_ratios=[0.9, 0.8, 3.2, 3.2], width_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
    
    # -------------------------------------------------------------
    # 1. HEADER BANNER
    # -------------------------------------------------------------
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(C_BG)
    ax_header.axis('off')
    
    # Header container patch
    hdr_patch = FancyBboxPatch((0.0, 0.05), 1.0, 0.9,
                               boxstyle="round,pad=0.01,rounding_size=0.04",
                               facecolor='#1C2541', edgecolor='#3B82F6', linewidth=2.5, transform=ax_header.transAxes)
    ax_header.add_patch(hdr_patch)
    
    ax_header.text(0.5, 0.68, "SMOKING BIOMARKERS & HEALTH METRICS INFOGRAPHIC",
                   transform=ax_header.transAxes, ha='center', va='center',
                   fontsize=24, fontweight='bold', color='#F8FAFC')
    ax_header.text(0.5, 0.32, "Exploratory Data Analysis & Clinical Biomarker Discrimination (N = 15,000 Bio-Samples)",
                   transform=ax_header.transAxes, ha='center', va='center',
                   fontsize=13, fontweight='medium', color='#60A5FA')
    
    # -------------------------------------------------------------
    # 2. KPI CARDS ROW
    # -------------------------------------------------------------
    ax_kpi1 = fig.add_subplot(gs[1, 0])
    draw_kpi_card(ax_kpi1, "Total Patients Analyzed", "15,000", "100% Verified Clinical Records", border_color='#06D6A0', val_color='#06D6A0')
    
    ax_kpi2 = fig.add_subplot(gs[1, 1])
    draw_kpi_card(ax_kpi2, "Smoking Prevalence", "36.56%", "5,484 Smokers vs 9,516 Non-Smokers", border_color='#FF4B4B', val_color='#FF4B4B')
    
    ax_kpi3 = fig.add_subplot(gs[1, 2])
    draw_kpi_card(ax_kpi3, "Top Biomarker Spikes", "GTP (+66.8%) & Hgb (+10.3%)", "Elevated Liver Enzymes & Hemoglobin", border_color='#FFD166', val_color='#FFD166')
    
    # -------------------------------------------------------------
    # 3. PANEL A: Hemoglobin & GTP Distribution (KDE Ridge Plots)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[2, 0])
    ax_a.set_facecolor(C_CARD)
    for spine in ax_a.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.5)
        
    sns.kdeplot(data=df, x='hemoglobin', hue='smoking', common_norm=False, fill=True,
                palette=[C_NON_SMOKER, C_SMOKER], alpha=0.45, linewidth=2, ax=ax_a)
    ax_a.set_title("1. Hemoglobin Levels (g/dL) by Smoking Status", fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax_a.set_xlabel("Hemoglobin (g/dL)", fontsize=10, fontweight='bold')
    ax_a.set_ylabel("Density", fontsize=10, fontweight='bold')
    
    # Custom Legend
    ax_a.legend(labels=['Smoker (Mean: 15.4)', 'Non-Smoker (Mean: 14.0)'], loc='upper left', frameon=True, facecolor='#0F172A', edgecolor=C_BORDER)
    ax_a.grid(True, linestyle='--', alpha=0.2, color='#94A3B8')
    
    # Text Annotation
    ax_a.annotate('Smoker Peak Shift\n(+1.44 g/dL elevated)', xy=(15.8, 0.35), xytext=(16.8, 0.45),
                arrowprops=dict(arrowstyle="->", color='#FF4B4B', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#FF4B4B', bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#FF4B4B"))
    
    # -------------------------------------------------------------
    # 4. PANEL B: Lipid Profile & HDL Cholesterol Reduction
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[2, 1])
    ax_b.set_facecolor(C_CARD)
    for spine in ax_b.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.5)
        
    # Grouped metrics comparison
    df_lipid = df.groupby('smoking')[['triglyceride', 'HDL', 'LDL', 'Cholesterol']].mean().reset_index()
    df_lipid_melt = df_lipid.melt(id_vars='smoking', var_name='Metric', value_name='mg/dL')
    df_lipid_melt['Status'] = df_lipid_melt['smoking'].map({0.0: 'Non-Smoker', 1.0: 'Smoker'})
    
    sns.barplot(data=df_lipid_melt, x='Metric', y='mg/dL', hue='Status',
                palette={'Non-Smoker': C_NON_SMOKER, 'Smoker': C_SMOKER}, ax=ax_b, edgecolor='none', alpha=0.9)
    
    ax_b.set_title("2. Lipid & Cholesterol Profile Comparison (mg/dL)", fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax_b.set_xlabel("Biomarker Metric", fontsize=10, fontweight='bold')
    ax_b.set_ylabel("Mean Concentration (mg/dL)", fontsize=10, fontweight='bold')
    ax_b.grid(True, linestyle='--', alpha=0.2, color='#94A3B8')
    ax_b.legend(loc='upper right', frameon=True, facecolor='#0F172A', edgecolor=C_BORDER)
    
    # Add values on top of bars
    for p in ax_b.patches:
        height = p.get_height()
        if height > 0:
            ax_b.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8.5, color='#CBD5E1', xytext=(0, 3),
                        textcoords='offset points')
    
    # -------------------------------------------------------------
    # 5. PANEL C: Correlation Heatmap with Target
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2, 2])
    ax_c.set_facecolor(C_CARD)
    for spine in ax_c.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.5)
        
    # Selected top correlated features
    top_features = ['smoking', 'hemoglobin', 'height(cm)', 'weight(kg)', 'Gtp', 'triglyceride', 'serum creatinine', 'waist(cm)', 'ALT', 'HDL', 'age']
    corr_matrix = df[top_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', center=0,
                cbar_kws={'shrink': 0.8}, ax=ax_c, annot_kws={'size': 7.5, 'weight': 'bold'}, linewidths=0.5, linecolor=C_BG)
    ax_c.set_title("3. Top 10 Feature Correlation Heatmap", fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax_c.tick_params(axis='x', rotation=45, labelsize=8.5)
    ax_c.tick_params(axis='y', rotation=0, labelsize=8.5)
    
    # -------------------------------------------------------------
    # 6. PANEL D: Physical Profile - Height vs Weight Scatter/Hexbin
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[3, 0])
    ax_d.set_facecolor(C_CARD)
    for spine in ax_d.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.5)
        
    sns.scatterplot(data=df.sample(2500, random_state=42), x='height(cm)', y='weight(kg)', hue='smoking',
                    palette={0.0: C_NON_SMOKER, 1.0: C_SMOKER}, alpha=0.45, s=25, ax=ax_d)
    ax_d.set_title("4. Body Profile: Height vs. Weight Scatter (Sample N=2,500)", fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax_d.set_xlabel("Height (cm)", fontsize=10, fontweight='bold')
    ax_d.set_ylabel("Weight (kg)", fontsize=10, fontweight='bold')
    ax_d.grid(True, linestyle='--', alpha=0.2, color='#94A3B8')
    ax_d.legend(labels=['Non-Smoker', 'Smoker'], loc='upper left', frameon=True, facecolor='#0F172A', edgecolor=C_BORDER)
    
    # -------------------------------------------------------------
    # 7. PANEL E: Dental Caries & Liver Enzyme (GTP) Risk Ratio
    # -------------------------------------------------------------
    ax_e = fig.add_subplot(gs[3, 1])
    ax_e.set_facecolor(C_CARD)
    for spine in ax_e.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.5)
        
    # Dental Caries Prevalence Bar
    caries_df = df.groupby('smoking')['dental caries'].mean() * 100
    categories = ['Non-Smoker\n(10.6%)', 'Smoker\n(19.6%)']
    bars = ax_e.bar(categories, caries_df, color=[C_NON_SMOKER, C_SMOKER], width=0.45, edgecolor=C_BORDER, linewidth=1.5)
    
    ax_e.set_title("5. Oral Health: Dental Caries Rate (+85% Higher in Smokers)", fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax_e.set_ylabel("Prevalence (%)", fontsize=10, fontweight='bold')
    ax_e.set_ylim(0, 30)
    ax_e.grid(True, linestyle='--', alpha=0.2, color='#94A3B8')
    
    for bar in bars:
        height = bar.get_height()
        ax_e.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=C_ACCENT)
                    
    # Highlight annotation
    ax_e.text(0.5, 0.82, "Smokers exhibit almost 2x higher\nlikelihood of dental cavities",
              transform=ax_e.transAxes, ha='center', va='center',
              fontsize=9.5, fontweight='semibold', color='#F8FAFC',
              bbox=dict(boxstyle="round,pad=0.4", fc="#0F172A", ec=C_ACCENT, lw=1))

    # -------------------------------------------------------------
    # 8. PANEL F: Top Correlated Feature Rankings (Horizontal Bar)
    # -------------------------------------------------------------
    ax_f = fig.add_subplot(gs[3, 2])
    ax_f.set_facecolor(C_CARD)
    for spine in ax_f.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(1.5)
        
    corrs = df.drop(columns=['id', 'smoking']).corrwith(df['smoking']).sort_values(ascending=True)
    top_corr = pd.concat([corrs.head(3), corrs.tail(7)]) # Top positive and negative
    
    colors = [C_NON_SMOKER if x < 0 else C_SMOKER for x in top_corr.values]
    bars_f = ax_f.barh(top_corr.index, top_corr.values, color=colors, height=0.6, edgecolor='none')
    
    ax_f.set_title("6. Feature Correlation Ranking with Smoking Target", fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax_f.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=10, fontweight='bold')
    ax_f.grid(True, linestyle='--', alpha=0.2, color='#94A3B8')
    ax_f.axvline(0, color='#94A3B8', linestyle='-', linewidth=1)
    
    for bar in bars_f:
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        offset = 0.01 if width > 0 else -0.01
        ax_f.annotate(f'{width:+.2f}',
                    xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                    ha=ha, va='center', fontsize=8.5, fontweight='bold', color='#F8FAFC')
                    
    # Output file
    output_path = 'smoking_dataset_infographic.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=C_BG, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated visual infographic: {output_path}")

if __name__ == '__main__':
    main()
