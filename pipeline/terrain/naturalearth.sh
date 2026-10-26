#!/usr/bin/env bash
# Descarrega as camadas Natural Earth 10m usadas pelo pipeline para .cache/naturalearth
set -euo pipefail
DIR="${NATURAL_EARTH_DIR:-$(dirname "$0")/../../.cache/naturalearth}"
mkdir -p "$DIR"
BASE=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
for f in ne_10m_admin_0_countries ne_10m_admin_1_states_provinces ne_10m_rivers_lake_centerlines ne_10m_rivers_europe; do
  [ -s "$DIR/$f.geojson" ] || curl -sSfL -o "$DIR/$f.geojson" "$BASE/$f.geojson"
done
