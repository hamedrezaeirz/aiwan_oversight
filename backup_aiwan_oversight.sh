#!/bin/bash
VPS_HOST="root@45.144.16.3"
VPS_PROJECT_DIR="~/aiwan_oversight"
LOCAL_BACKUP_ROOT="/home/hamedrezaeirz/Documents/aiwan_oversight_backup"
DATE_TAG=$(date +%F)

DEST="$LOCAL_BACKUP_ROOT/$DATE_TAG"
mkdir -p "$DEST"

echo "Backing up results/ and logs/ from $VPS_HOST to $DEST ..."

rsync -avz --progress "$VPS_HOST:$VPS_PROJECT_DIR/results/" "$DEST/results/"
rsync -avz --progress "$VPS_HOST:$VPS_PROJECT_DIR/logs/" "$DEST/logs/"

echo ""
echo "Done. Backed up to: $DEST"
echo "Row counts in this backup's per-seed CSVs (for a quick progress glance):"
wc -l "$DEST"/results/results_oversight_seeds_*.csv 2>/dev/null || echo "  (no per-seed CSVs found yet -- may be too early in the sweep)"
