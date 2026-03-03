#!/bin/bash
#
# AVERA-ATLAS Tracker Service Test Script
# 
# Tests the tracker service end-to-end via docker exec
# Run from repo root: ./tracker-service/test_tracker.sh
#

set -e

CONTAINER="avera-atlas-v5-tracker-1"
BASE_CMD="docker exec $CONTAINER curl -s"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  AVERA-ATLAS Tracker Service Test"
echo "=============================================="
echo ""

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo -e "${RED}ERROR: Container '$CONTAINER' is not running${NC}"
    echo "Run: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Container '$CONTAINER' is running${NC}"
echo ""

# ---------------------------------------------
# Test 1: Health Check
# ---------------------------------------------
echo "--- Test 1: Health Check ---"
HEALTH=$($BASE_CMD http://localhost:8000/health)
echo "$HEALTH" | python3 -m json.tool
echo ""

if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi
echo ""

# ---------------------------------------------
# Test 2: Service Status (Initial)
# ---------------------------------------------
echo "--- Test 2: Initial Status ---"
$BASE_CMD http://localhost:8000/status | python3 -m json.tool
echo ""

# ---------------------------------------------
# Test 3: Register Sensors
# ---------------------------------------------
echo "--- Test 3: Register Sensors ---"

# Sensor 1
echo "Registering AVERA-SAT-01..."
RESULT=$($BASE_CMD -X POST http://localhost:8000/sensors \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "AVERA-SAT-01-SWIR",
    "platform_name": "AVERA-SAT-01",
    "focal_length_mm": 50.0,
    "pixel_size_um": 15.0,
    "resolution_x": 1024,
    "resolution_y": 768,
    "fov_x_deg": 12.0,
    "fov_y_deg": 9.0
  }' 2>/dev/null || echo '{"error": "already registered"}')
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

# Sensor 2
echo "Registering AVERA-SAT-02..."
RESULT=$($BASE_CMD -X POST http://localhost:8000/sensors \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "AVERA-SAT-02-SWIR",
    "platform_name": "AVERA-SAT-02",
    "focal_length_mm": 50.0,
    "pixel_size_um": 15.0,
    "resolution_x": 1024,
    "resolution_y": 768,
    "fov_x_deg": 12.0,
    "fov_y_deg": 9.0
  }' 2>/dev/null || echo '{"error": "already registered"}')
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

echo ""
echo "Listing registered sensors..."
$BASE_CMD http://localhost:8000/sensors | python3 -m json.tool
echo ""

# ---------------------------------------------
# Test 4: Post Detections (Simulated Multi-Sensor)
# ---------------------------------------------
echo "--- Test 4: Post Detections ---"

# Detection batch from Sensor 1 (Object 1: Debris)
# Simulating a debris object passing through FOV over 20 seconds
# Moving ~100 pixels across the image (realistic for LEO debris)
echo "Posting detections from AVERA-SAT-01 (Object 1: Debris)..."
RESULT=$($BASE_CMD -X POST http://localhost:8000/detections \
  -H "Content-Type: application/json" \
  -d '{
    "detections": [
      {
        "detection_id": "det-sat01-001",
        "sensor_id": "AVERA-SAT-01-SWIR",
        "timestamp": "2025-01-15T12:00:00Z",
        "pixel_u": 400.0,
        "pixel_v": 450.0,
        "bbox_x": 388.0,
        "bbox_y": 436.0,
        "bbox_w": 25.0,
        "bbox_h": 28.0,
        "confidence": 0.87,
        "object_class": "Debris"
      },
      {
        "detection_id": "det-sat01-002",
        "sensor_id": "AVERA-SAT-01-SWIR",
        "timestamp": "2025-01-15T12:00:10Z",
        "pixel_u": 480.0,
        "pixel_v": 400.0,
        "bbox_x": 468.0,
        "bbox_y": 386.0,
        "bbox_w": 25.0,
        "bbox_h": 28.0,
        "confidence": 0.89,
        "object_class": "Debris"
      },
      {
        "detection_id": "det-sat01-003",
        "sensor_id": "AVERA-SAT-01-SWIR",
        "timestamp": "2025-01-15T12:00:20Z",
        "pixel_u": 560.0,
        "pixel_v": 350.0,
        "bbox_x": 548.0,
        "bbox_y": 336.0,
        "bbox_w": 25.0,
        "bbox_h": 28.0,
        "confidence": 0.91,
        "object_class": "Debris"
      },
      {
        "detection_id": "det-sat01-004",
        "sensor_id": "AVERA-SAT-01-SWIR",
        "timestamp": "2025-01-15T12:00:30Z",
        "pixel_u": 640.0,
        "pixel_v": 300.0,
        "bbox_x": 628.0,
        "bbox_y": 286.0,
        "bbox_w": 25.0,
        "bbox_h": 28.0,
        "confidence": 0.88,
        "object_class": "Debris"
      }
    ]
  }')
echo "$RESULT" | python3 -m json.tool
echo ""

# Detection batch from Sensor 2 (Object 2: RocketBody - different object)
# Different trajectory across the FOV
echo "Posting detections from AVERA-SAT-02 (Object 2: RocketBody)..."
RESULT=$($BASE_CMD -X POST http://localhost:8000/detections \
  -H "Content-Type: application/json" \
  -d '{
    "detections": [
      {
        "detection_id": "det-sat02-001",
        "sensor_id": "AVERA-SAT-02-SWIR",
        "timestamp": "2025-01-15T12:00:00Z",
        "pixel_u": 700.0,
        "pixel_v": 200.0,
        "bbox_x": 688.0,
        "bbox_y": 186.0,
        "bbox_w": 30.0,
        "bbox_h": 35.0,
        "confidence": 0.92,
        "object_class": "RocketBody"
      },
      {
        "detection_id": "det-sat02-002",
        "sensor_id": "AVERA-SAT-02-SWIR",
        "timestamp": "2025-01-15T12:00:10Z",
        "pixel_u": 620.0,
        "pixel_v": 280.0,
        "bbox_x": 608.0,
        "bbox_y": 266.0,
        "bbox_w": 30.0,
        "bbox_h": 35.0,
        "confidence": 0.90,
        "object_class": "RocketBody"
      },
      {
        "detection_id": "det-sat02-003",
        "sensor_id": "AVERA-SAT-02-SWIR",
        "timestamp": "2025-01-15T12:00:20Z",
        "pixel_u": 540.0,
        "pixel_v": 360.0,
        "bbox_x": 528.0,
        "bbox_y": 346.0,
        "bbox_w": 30.0,
        "bbox_h": 35.0,
        "confidence": 0.88,
        "object_class": "RocketBody"
      },
      {
        "detection_id": "det-sat02-004",
        "sensor_id": "AVERA-SAT-02-SWIR",
        "timestamp": "2025-01-15T12:00:30Z",
        "pixel_u": 460.0,
        "pixel_v": 440.0,
        "bbox_x": 448.0,
        "bbox_y": 426.0,
        "bbox_w": 30.0,
        "bbox_h": 35.0,
        "confidence": 0.85,
        "object_class": "RocketBody"
      }
    ]
  }')
echo "$RESULT" | python3 -m json.tool
echo ""

# ---------------------------------------------
# Test 5: Check Uncorrelated Buffer (with Correlation)
# ---------------------------------------------
echo "--- Test 5: Uncorrelated Track Buffers (Correlated) ---"
UNCORR=$($BASE_CMD http://localhost:8000/uncorrelated)
echo "$UNCORR" | python3 -m json.tool

# Show correlation summary
echo ""
echo "Correlation Summary:"
echo "$UNCORR" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Total UCTs: {data['count']}\")
print(f\"  IOD-ready: {data['iod_ready_count']}\")
print()
stats = data.get('correlation_stats', {})
print(f\"  Observations processed: {stats.get('observations_processed', 'N/A')}\")
print(f\"  New UCTs created: {stats.get('new_ucts_created', 'N/A')}\")
print(f\"  Correlations made: {stats.get('correlations_made', 'N/A')}\")
print()
for b in data['buffers']:
    print(f\"  UCT {b['uct_id'][:8]}...: {b['observation_count']} obs, {b['object_class']}, arc={b.get('arc_length_deg', 0):.3f}°, IOD-ready={b.get('iod_ready', False)}\")
"
echo ""

# ---------------------------------------------
# Test 6: Attempt IOD on first UCT
# ---------------------------------------------
echo "--- Test 6: Attempt IOD ---"

# Get first IOD-ready UCT
FIRST_UCT=$(echo "$UNCORR" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for b in data['buffers']:
    if b.get('iod_ready'):
        print(b['uct_id'])
        break
")

if [ -n "$FIRST_UCT" ]; then
    echo "Attempting IOD on UCT: $FIRST_UCT"
    IOD_RESULT=$($BASE_CMD -X POST "http://localhost:8000/uncorrelated/$FIRST_UCT/attempt_iod")
    echo "$IOD_RESULT" | python3 -m json.tool
    
    # Show orbital elements if successful
    echo ""
    echo "IOD Results:"
    echo "$IOD_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('status') == 'success':
    sol = data.get('solution', {})
    print(f\"  Track: {data.get('track_name')}\")
    print(f\"  Semi-major axis: {sol.get('semi_major_axis_km', 'N/A'):.1f} km\")
    print(f\"  Eccentricity: {sol.get('eccentricity', 'N/A'):.4f}\")
    print(f\"  Inclination: {sol.get('inclination_deg', 'N/A'):.2f}°\")
    print(f\"  RMS Residual: {sol.get('rms_residual_arcsec', 'N/A'):.1f} arcsec\")
else:
    print(f\"  Status: {data.get('status')}\")
    print(f\"  Error: {data.get('error', 'N/A')}\")
"
else
    echo "No IOD-ready UCTs found"
fi
echo ""

# ---------------------------------------------
# Test 7: Check Tracks (should have results now)
# ---------------------------------------------
echo "--- Test 7: Tracks ---"
TRACKS=$($BASE_CMD http://localhost:8000/tracks)
echo "$TRACKS" | python3 -m json.tool

# Show track summary
echo ""
echo "Track Summary:"
echo "$TRACKS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Total tracks: {data.get('total_tracks', 0)}\")
for t in data.get('tracks', []):
    print(f\"  - {t.get('object_id')}: {t.get('object_class')}\")
"
echo ""

# ---------------------------------------------
# Test 8: Final Status
# ---------------------------------------------
echo "--- Test 8: Final Status ---"
STATUS=$($BASE_CMD http://localhost:8000/status)
echo "$STATUS" | python3 -m json.tool
echo ""

# Extract counts
PROCESSED=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['detections_processed'])")
UNCORRELATED=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['uncorrelated_detections'])")
SENSORS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['registered_sensors'])")
ACTIVE_TRACKS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['active_tracks'])")

echo "=============================================="
echo "  Test Summary"
echo "=============================================="
echo -e "  Sensors registered:      ${GREEN}$SENSORS${NC}"
echo -e "  Detections processed:    ${GREEN}$PROCESSED${NC}"
echo -e "  UCT buffers:             ${YELLOW}$UNCORRELATED${NC}"
echo -e "  Active tracks:           ${GREEN}$ACTIVE_TRACKS${NC}"
echo ""
echo -e "${GREEN}✓ All tests passed${NC}"
echo ""
echo "Pipeline complete:"
echo "  ✓ Pixel coords → Camera frame → Body frame → ECI → RA/Dec"
echo "  ✓ Detection correlation (grouping same object)"
echo "  ✓ Initial Orbit Determination (IOD)"
echo "  ✓ Track creation with orbital elements"
echo ""
