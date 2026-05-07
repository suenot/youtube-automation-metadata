#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_DIR/input"
OUTPUT_DIR="$PROJECT_DIR/output"
TEMP_DIR="$PROJECT_DIR/temp"

VENV_PYTHON="${PROJECT_DIR}/../video_maker/venv/bin/python3"

# Video → slug mapping
declare -A VIDEO_MAP
VIDEO_MAP["The_Fill_Ambiguity_Trap__Engineering_an_Adaptive_Drill-Down_Bac.mp4"]="adaptive-resolution-drill-down-backtest"
VIDEO_MAP["The_Overfitting_Trap__Architecting_Walk-Forward_Optimization.mp4"]="walk-forward-optimization"
VIDEO_MAP["Architecting_the_High-Frequency_Quant_Engine.mp4"]="data-communication-algotrading"
VIDEO_MAP["The_Single-Point_Illusion__Engineering_Robust_Backtests.mp4"]="monte-carlo-bootstrap-backtest"
VIDEO_MAP["0ns_Encode,_30ns_IPC__Deconstructing_ZigBolt_s_Hot_Path.mp4"]="zigbolt-zig-messaging-hft"

for video_file in "${!VIDEO_MAP[@]}"; do
    slug="${VIDEO_MAP[$video_file]}"
    video_path="$INPUT_DIR/$video_file"
    out_dir="$OUTPUT_DIR/$slug"

    if [ ! -f "$video_path" ]; then
        echo "SKIP: $video_file not found"
        continue
    fi

    echo "========================================"
    echo "Processing: $video_file"
    echo "Slug: $slug"
    echo "Output: $out_dir"
    echo "========================================"

    "$VENV_PYTHON" "$SCRIPT_DIR/prepare_metadata.py" \
        --video "$video_path" \
        --slug "$slug" \
        --output-dir "$out_dir" \
        --temp-dir "$TEMP_DIR/$slug" \
        --whisper-model base

    # Move video from input to output folder
    mv "$video_path" "$out_dir/"
    echo "Moved $video_file -> $out_dir/"

    echo ""
done

echo "All done!"
