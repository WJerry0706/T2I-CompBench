#!/bin/bash

# Define the output file for the scores
RESULTS_FILE="evaluation_scores.txt"

# Get the current date and time
DATETIME=$(date)

# Initialize the results file with a timestamp for this run
echo "Evaluation run at: $DATETIME" > "$RESULTS_FILE"
echo "-------------------------------------" >> "$RESULTS_FILE"

# --- 1. Run BLIP-VQA Evaluation ---
echo "Running BLIP-VQA Evaluation..."
blip_output=$(bash BLIPvqa_eval/test.sh)
blip_score=$(echo "$blip_output" | grep "BLIP-VQA score:" | awk '{print $3}')
echo "BLIP-VQA Score: $blip_score" >> "$RESULTS_FILE"
echo "BLIP-VQA evaluation complete."

# --- 2. Run CLIPScore Evaluation ---
echo "Running CLIPScore Evaluation..."
clip_output=$(bash CLIPScore_eval/test.sh)
clip_score=$(echo "$clip_output" | grep "score avg:" | awk '{print $3}')
echo "CLIPScore: $clip_score" >> "$RESULTS_FILE"
echo "CLIPScore evaluation complete."

# # --- 3. Run UniDet 2D Spatial Evaluation ---
# echo "Running UniDet 2D Spatial Evaluation..."
# # Change directory as required by the script
# cd UniDet_eval/
# unidet_output=$(python 2D_spatial_eval.py)
# cd .. # IMPORTANT: Go back to the root directory
# unidet_score=$(echo "$unidet_output" | grep "avg score:" | awk '{print $3}')
# echo "UniDet 2D Spatial Score: $unidet_score" >> "$RESULTS_FILE"
# echo "UniDet 2D Spatial evaluation complete."

# --- 4. Run 3-in-1 Evaluation ---
echo "Running 3-in-1 Evaluation..."
three_in_one_output=$(bash 3_in_1_eval/test.sh)
three_in_one_score=$(echo "$three_in_one_output" | grep "avg score:" | awk '{print $3}')
echo "3-in-1 Score: $three_in_one_score" >> "$RESULTS_FILE"
echo "3-in-1 evaluation complete."

# --- Final Message ---
echo "-------------------------------------"
echo "All evaluations are complete."
echo "Final scores have been saved to $RESULTS_FILE"
cat "$RESULTS_FILE"