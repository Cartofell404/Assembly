#!/bin/sh
# End-to-end demo of the Rust CLI, with no models to download.
#
# Creates GGUF-shaped files (a valid magic/version/count header, zero tensors,
# no metadata), writes a config that points at them, then runs the real binary
# through validate -> build -> info -> info --verify.
#
# These are not usable models. Nothing infers from them; they exist so the
# format, the pre-flight checks and the writer can be exercised without pulling
# gigabytes off the network. Real routing needs real weights and an inference
# engine, which this demo deliberately avoids.
#
#   sh demo.sh            # writes into target/demo
#   sh demo.sh /tmp/where  # or wherever you want it

set -eu

here=$(cd "$(dirname "$0")" && pwd)
out=${1:-"$here/target/demo"}
bin="$here/target/debug/assembly"

cargo build -p assembly --quiet

# A GGUF with a header the validator accepts and no content worth loading.
fake_gguf() {
    printf 'GGUF' > "$1"
    printf '\003\000\000\000' >> "$1"   # version 3
    printf '\000\000\000\000\000\000\000\000' >> "$1"   # tensor_count
    printf '\000\000\000\000\000\000\000\000' >> "$1"   # metadata_count
    dd if=/dev/zero bs=1024 count="$2" >> "$1" 2>/dev/null
}

rm -rf "$out"
mkdir -p "$out"

fake_gguf "$out/route.gguf" 64
fake_gguf "$out/general.gguf" 512
fake_gguf "$out/code.gguf" 1024
fake_gguf "$out/small.gguf" 128

cat > "$out/config.json" <<EOF
{
  "assembly_name": "Demo",
  "persona": "Be brief, then stop.",
  "embedding": {
    "name": "route",
    "path": "$out/route.gguf",
    "confidence_threshold": 0.6,
    "fallback_model": "general"
  },
  "models": [
    {"name": "general", "path": "$out/general.gguf", "anchors": ["how are you", "what is the weather"]},
    {"name": "code", "path": "$out/code.gguf", "anchors": ["write a function", "fix this bug"]},
    {"name": "small", "path": "$out/small.gguf", "anchors": ["hi", "thanks", "ok"]}
  ]
}
EOF

echo "==> validate"
"$bin" validate "$out/config.json"

echo
echo "==> build"
"$bin" build "$out/config.json" -o "$out/demo.asmb"

echo
echo "==> info"
"$bin" info "$out/demo.asmb"

echo
echo "==> info --verify"
"$bin" info "$out/demo.asmb" --verify

echo
echo "archive: $out/demo.asmb"