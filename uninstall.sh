#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[-]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

CARGO_HOME="${HOME}/.cargo"
BINDIR="${HOME}/.local/bin"

log "Uninstalling Assembly..."
cargo uninstall assembly 2>/dev/null || warn "cargo uninstall failed (may not be installed)"
rm -f "${BINDIR}/assembly" 2>/dev/null || true

log "Assembly uninstalled successfully"
