#!/usr/bin/env bash
# smoke_test_pipeline.sh — SCRUM-325 detector→tracker smoke test
#
# Starts ingest, detector, tracker via docker compose, posts a synthetic
# image to detector /predict, then confirms tracker received the detection.
# Exits 0 on success, non-zero with diagnostics on failure.

set -euo pipefail

DETECTOR_URL="http://localhost:8000"
TRACKER_URL="http://localhost:8002"
TIMEOUT=10

# ── helpers ──────────────────────────────────────────────────────────────

log()  { echo "[smoke] $*"; }
fail() { echo "[FAIL] $*" >&2; dump_logs; exit 1; }

dump_logs() {
    log "── detector logs (last 50) ──"
    docker compose logs --tail=50 detector 2>/dev/null || true
    log "── tracker logs (last 50) ──"
    docker compose logs --tail=50 tracker 2>/dev/null || true
    log "── ingest logs (last 50) ──"
    docker compose logs --tail=50 ingest 2>/dev/null || true
}

wait_for_health() {
    local url="$1" name="$2"
    local elapsed=0
    while [ $elapsed -lt $TIMEOUT ]; do
        if curl -sf "$url/health" > /dev/null 2>&1; then
            log "$name healthy"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "$name did not become healthy within ${TIMEOUT}s"
}

# ── start services ───────────────────────────────────────────────────────

log "Starting services..."
docker compose up -d ingest detector tracker

wait_for_health "$DETECTOR_URL" "detector"
wait_for_health "$TRACKER_URL" "tracker"

# ── generate test image ─────────────────────────────────────────────────

# 256x256 black PNG with a small white square in the centre
IMG_FILE=$(mktemp /tmp/smoke_img_XXXX.png)
python3 -c "
from PIL import Image
img = Image.new('RGB', (256, 256), (0, 0, 0))
for x in range(120, 136):
    for y in range(120, 136):
        img.putpixel((x, y), (255, 255, 255))
img.save('$IMG_FILE')
"
B64=$(python3 -c "import base64,sys; print(base64.b64encode(open('$IMG_FILE','rb').read()).decode())")
rm -f "$IMG_FILE"

# ── POST to detector ────────────────────────────────────────────────────

log "Posting image to detector /predict..."
DETECT_RESP=$(curl -sf -X POST "$DETECTOR_URL/predict" \
    -H "Content-Type: application/json" \
    -d "{\"base64_data\":\"$B64\",\"frame_id\":\"smoke-1\"}") \
    || fail "Detector /predict returned non-200"

log "Detector response: $DETECT_RESP"

# ── poll tracker ─────────────────────────────────────────────────────────

log "Polling tracker for detections..."
elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    TRACKER_RESP=$(curl -sf "$TRACKER_URL/status" 2>/dev/null) || true
    PROCESSED=$(echo "$TRACKER_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('detections_processed',0))" 2>/dev/null || echo 0)
    if [ "$PROCESSED" -gt 0 ] 2>/dev/null; then
        log "Tracker has processed $PROCESSED detection(s)"
        log "SMOKE TEST PASSED"
        exit 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

# Also check /uncorrelated as a secondary signal
UCT_RESP=$(curl -sf "$TRACKER_URL/uncorrelated" 2>/dev/null) || true
UCT_COUNT=$(echo "$UCT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null || echo 0)
if [ "$UCT_COUNT" -gt 0 ] 2>/dev/null; then
    log "Tracker has $UCT_COUNT UCT(s)"
    log "SMOKE TEST PASSED"
    exit 0
fi

fail "No detections reached tracker within ${TIMEOUT}s"
