#!/usr/bin/env bash
set -euo pipefail

# Assembly - Complete installer that sets up everything needed
# Installs Rust, build dependencies, builds and installs assembly globally

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $*"; }
info() { echo -e "${BLUE}[i]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err() { echo -e "${RED}[x]${NC} $*"; }
bold() { echo -e "${BLUE}${*}${NC}"; }

# Get repo directory
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_CRATES_DIR="$REPO_DIR/rust"
INSTALL_PREFIX="${HOME}/.local"
BINDIR="${INSTALL_PREFIX}/bin"
CARGO_HOME="${HOME}/.cargo"
CARGO_BIN="${CARGO_HOME}/bin"

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

ensure_dir() {
    mkdir -p "$1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_rust() {
    if command_exists rustc && command_exists cargo; then
        local rust_ver
        rust_ver="$(rustc --version)"
        log "Rust already installed: ${rust_ver}"
        return 0
    fi
    
    log "Rust not found. Installing Rust via rustup (no sudo required)..."
    info "This will install rustup to ${CARGO_HOME}"
    
    # Download and run rustup installer
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    
    # Source rust environment
    # shellcheck disable=SC1090
    source "${CARGO_HOME}/env"
    
    log "Rust installed successfully"
}

install_deps_linux() {
    local pkg_manager=""
    local pkgs=""
    
    if command_exists apt-get; then
        pkg_manager="apt-get"
        # cmake, clang, pkg-config, libssl-dev, build-essential
        pkgs="build-essential cmake clang pkg-config libssl-dev"
    elif command_exists yum; then
        pkg_manager="yum"
        pkgs="cmake clang pkgconfig openssl-devel gcc-c++ make"
    elif command_exists dnf; then
        pkg_manager="dnf"
        pkgs="cmake clang pkgconf-pkg-config openssl-devel gcc-c++ make"
    elif command_exists pacman; then
        pkg_manager="pacman"
        pkgs="cmake clang pkgconf openssl base-devel"
    elif command_exists apk; then
        pkg_manager="apk"
        pkgs="cmake clang pkgconf openssl-dev build-base"
    else
        warn "Unknown package manager. Attempting to build without installing system deps."
        warn "You may need to install cmake, clang, openssl-dev manually."
        return 0
    fi
    
    info "Detected ${pkg_manager}, installing build dependencies: ${pkgs}"
    
    if [[ "${pkg_manager}" == "apt-get" ]]; then
        sudo apt-get update -qq 2>/dev/null || true
        sudo apt-get install -y -qq ${pkgs} 2>/dev/null || warn "Some packages failed to install (may already be present)"
    elif [[ "${pkg_manager}" == "pacman" ]]; then
        sudo pacman -Sy --noconfirm --needed ${pkgs} 2>/dev/null || warn "Some packages failed to install"
    else
        sudo ${pkg_manager} install -y ${pkgs} 2>/dev/null || warn "Some packages failed to install"
    fi
    
    log "System dependencies installed"
}

install_deps_macos() {
    if command_exists brew; then
        log "Homebrew detected, installing dependencies..."
        brew install --quiet cmake llvm pkg-config openssl 2>/dev/null || warn "Some packages failed"
    else
        info "Homebrew not found. Checking for Xcode Command Line Tools..."
        if ! xcode-select -p >/dev/null 2>&1; then
            warn "Xcode Command Line Tools not found"
            info "Run 'xcode-select --install' and re-run this installer"
        fi
        info "CMake and clang should come with Xcode CLT"
    fi
}

install_deps() {
    case "${OS}" in
        Linux*)
            install_deps_linux
            ;;
        Darwin*)
            install_deps_macos
            ;;
        *)
            warn "Unsupported OS: ${OS}. Proceeding anyway..."
            ;;
    esac
}

build_and_install() {
    log "Building and installing Assembly (this may take 5-15 minutes on first build)..."
    
    # Ensure cargo is in PATH
    # shellcheck disable=SC1090
    [[ -f "${CARGO_HOME}/env" ]] && source "${CARGO_HOME}/env"
    
    cd "${RUST_CRATES_DIR}"
    
    # Install with locked deps for reproducibility
    cargo install --path crates/cli --locked --force
    
    log "Assembly built and installed successfully"
}

setup_path() {
    ensure_dir "${BINDIR}"
    
    # Create symlink to cargo bin if needed
    local installed_bin="${CARGO_BIN}/assembly"
    local symlink_bin="${BINDIR}/assembly"
    
    if [[ -f "${installed_bin}" ]]; then
        if [[ ! -L "${symlink_bin}" ]] && [[ ! -f "${symlink_bin}" ]]; then
            ln -s "${installed_bin}" "${symlink_bin}" 2>/dev/null || true
        fi
    fi
    
    # Check if in PATH
    if [[ ":${PATH}:" != *":${BINDIR}:"* ]] && [[ ":${PATH}:" != *":${CARGO_BIN}:"* ]]; then
        warn "${CARGO_BIN} and ${BINDIR} are not in your PATH"
        echo
        bold "To add to your PATH, add this to your shell config (~/.bashrc, ~/.zshrc, etc.):"
        echo "  export PATH=\"${CARGO_BIN}:\$HOME/.local/bin:\$PATH\""
        echo
        info "Or run: source ${CARGO_HOME}/env"
    fi
}

verify_install() {
    # shellcheck disable=SC1090
    [[ -f "${CARGO_HOME}/env" ]] && source "${CARGO_HOME}/env"
    
    if command_exists assembly; then
        local ver
        ver="$(assembly --version 2>/dev/null || echo "installed")"
        log "✓ Assembly ${ver} is installed and ready"
        info "Location: $(command -v assembly)"
        return 0
    else
        err "Installation verification failed"
        return 1
    fi
}

create_uninstaller() {
    local uninstall_script="${REPO_DIR}/uninstall.sh"
    
    cat > "${uninstall_script}" << 'UNINSTALL'
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
UNINSTALL
    chmod +x "${uninstall_script}"
    log "Created uninstaller: uninstall.sh"
}

main() {
    bold "=== Assembly Complete Installer ==="
    echo
    info "OS: ${OS} (${ARCH})"
    info "Installing to: ${CARGO_BIN} and ${BINDIR}"
    echo
    
    ensure_dir "${INSTALL_PREFIX}"
    ensure_dir "${BINDIR}"
    
    install_deps
    install_rust
    build_and_install
    setup_path
    create_uninstaller
    verify_install
    
    echo
    bold "=== Installation Complete ==="
    log "Run 'assembly --help' to get started"
    log "Run './uninstall.sh' to remove Assembly"
}

main "$@"
