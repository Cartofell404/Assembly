#!/bin/bash
set -euo pipefail

# Release script for Assembly
# Usage: ./scripts/release.sh <version> [--dry-run]

VERSION=${1:-}
DRY_RUN=${2:-}

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUST_DIR="${REPO_ROOT}/rust"

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version> [--dry-run]"
    echo "Example: $0 v0.1.0"
    exit 1
fi

if [[ ! $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-beta\.[0-9]+|-rc\.[0-9]+)?$ ]]; then
    echo "Error: Version must match semver (e.g., v0.1.0, v0.1.0-beta.1, v1.0.0-rc.1)"
    exit 1
fi

echo "=== Assembly Release Script ==="
echo "Version: $VERSION"
echo "Dry run: ${DRY_RUN:-false}"
echo ""

# Check we're on main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: Must be on main branch (currently on $BRANCH)"
    exit 1
fi

# Check working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: Working directory not clean. Commit or stash changes first."
    git status --short
    exit 1
fi

# Run tests
echo "Running tests..."
(cd "${RUST_DIR}" && cargo test --workspace --locked)

# Build release
echo "Building release..."
(cd "${RUST_DIR}" && cargo build --release --locked --bin assembly)

# Test the binary
echo "Testing binary..."
"${RUST_DIR}/target/release/assembly" --version

# Create tag
TAG="$VERSION"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo "Creating tag $TAG..."
    git tag -a "$TAG" -m "Release $VERSION"
    git push origin "$TAG"
    echo "Tag pushed. GitHub Actions will build and create release."
else
    echo "[DRY RUN] Would create tag $TAG and push to origin"
fi

echo ""
echo "=== Release Complete ==="
echo "Version: $VERSION"
echo "Tag: $TAG"
echo ""
echo "Next steps:"
echo "1. GitHub Actions will build binaries for Linux/macOS/Windows"
echo "2. Release will be created with binaries and checksums"
echo "3. Verify release at: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/releases/tag/$VERSION"
