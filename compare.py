import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# --- Configuration ---
# Input JSON files
entropy_file = 'examples/entropy_results.json'
blip_scores_file = 'examples/annotation_blip/vqa_result.json'
clip_scores_file = 'examples/annotation_clip/vqa_result.json'

# Directory where the output graphs will be saved
output_dir = 'temp_1.0'

try:
    # --- Part 1: Merge Data Sources ---
    print("--- Merging data files... ---")
    
    # Load and prepare Entropy data
    df_entropy = pd.read_json(entropy_file)
    df_entropy['filename'] = df_entropy['image_path'].apply(os.path.basename)
    
    # NEW: Scale the entropy values immediately after reading the data
    df_entropy['entropy'] = pd.to_numeric(df_entropy['entropy'], errors='coerce')
    df_entropy['entropy'] = df_entropy['entropy'] / 256
    print("✅ Entropy data loaded and scaled by 256.")

    # Load and prepare BLIP Scores
    df_blip = pd.read_json(blip_scores_file)
    df_blip['filename'] = df_blip['image_path'].apply(os.path.basename)
    df_blip.rename(columns={'answer': 'blip_score'}, inplace=True)

    # Load and prepare CLIP Scores
    df_clip = pd.read_json(clip_scores_file)
    df_clip['filename'] = df_clip['image_path'].apply(os.path.basename)
    df_clip.rename(columns={'answer': 'clip_score'}, inplace=True)

    # Merge all three dataframes
    merged_df = pd.merge(
        df_entropy[['filename', 'entropy']],
        df_blip[['filename', 'blip_score']],
        on='filename', how='inner'
    )
    final_df = pd.merge(
        merged_df,
        df_clip[['filename', 'clip_score']],
        on='filename', how='inner'
    )
    print("✅ Merging complete.")

    # --- Part 2: Analyze and Plot Correlations ---
    print("\n--- Starting correlation analysis... ---")
    os.makedirs(output_dir, exist_ok=True)

    # Clean and prepare final data for analysis
    final_df['blip_score'] = pd.to_numeric(final_df['blip_score'], errors='coerce')
    final_df['clip_score'] = pd.to_numeric(final_df['clip_score'], errors='coerce')
    final_df.dropna(inplace=True)

    # 1. Analyze and Plot Entropy vs. BLIP Score
    corr_blip, p_blip = pearsonr(final_df['entropy'], final_df['blip_score'])
    print(f"Correlation (Entropy vs. BLIP Score): r={corr_blip:.4f}, p={p_blip:.4f}")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(final_df['entropy'], final_df['blip_score'], alpha=0.5, label='Data points')
    coeffs_blip = np.polyfit(final_df['entropy'], final_df['blip_score'], 1)
    p_blip = np.poly1d(coeffs_blip)
    x_line = np.linspace(final_df['entropy'].min(), final_df['entropy'].max(), 100)
    ax1.plot(x_line, p_blip(x_line), "r--", label="Linear Regression")
    
    ax1.set_title('Correlation between Scaled Entropy and BLIP Score', fontsize=16)
    ax1.set_xlabel('Scaled Entropy (Original / 256)', fontsize=12)
    ax1.set_ylabel('BLIP Score', fontsize=12)
    ax1.legend()
    
    plot1_path = os.path.join(output_dir, 'correlation_entropy_vs_blip_scaled.png')
    plt.savefig(plot1_path)
    print(f"✅ BLIP correlation plot saved to: '{plot1_path}'")
    plt.close(fig1)

    # 2. Analyze and Plot Entropy vs. CLIP Score
    corr_clip, p_clip = pearsonr(final_df['entropy'], final_df['clip_score'])
    print(f"Correlation (Entropy vs. CLIP Score): r={corr_clip:.4f}, p={p_clip:.4f}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(final_df['entropy'], final_df['clip_score'], alpha=0.5, color='green', label='Data points')
    coeffs_clip = np.polyfit(final_df['entropy'], final_df['clip_score'], 1)
    p_clip = np.poly1d(coeffs_clip)
    x_line = np.linspace(final_df['entropy'].min(), final_df['entropy'].max(), 100)
    ax2.plot(x_line, p_clip(x_line), "r--", label="Linear Regression")

    ax2.set_title('Correlation between Scaled Entropy and CLIP Score', fontsize=16)
    ax2.set_xlabel('Scaled Entropy (Original / 256)', fontsize=12)
    ax2.set_ylabel('CLIP Score', fontsize=12)
    ax2.legend()
    
    plot2_path = os.path.join(output_dir, 'correlation_entropy_vs_clip_scaled.png')
    plt.savefig(plot2_path)
    print(f"✅ CLIP correlation plot saved to: '{plot2_path}'")
    plt.close(fig2)

except FileNotFoundError as e:
    print(f"❌ Error: An input file was not found. Please check your paths.")
    print(f"Details: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")