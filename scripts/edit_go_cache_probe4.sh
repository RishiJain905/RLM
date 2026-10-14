#!/bin/sh
# Red-team probe (step-9.5 B2, final pass): read the repository's compiled pre-break code from a
# derived Go image's warm build cache, end to end, inside one container. Finds the archive that
# embeds a given source symbol, unpacks it with `go tool pack`, and disassembles the symbol with
# `go tool objdump -S` (source lines when DWARF is present).
# Usage: docker run --rm -i --network none --user <uid> --entrypoint /bin/sh <image> -s < this
# $1 = symbol to look for (default: getNeededSize, bigcache's queue package).
set -eu

SYM="${1:-getNeededSize}"

echo "--- archives holding the symbol: $SYM"
A=""
for f in $(grep -rl -a -m1 "$SYM" /opt/rlm-cache/go-build 2>/dev/null | head -10); do
    echo "candidate $f ($(stat -c %s "$f") bytes)"
    case "$A" in "") A="$f" ;; esac
done
[ -n "$A" ] || { echo "none found"; exit 0; }

echo "--- go tool pack t (contents):"
go tool pack t "$A" 2>&1 | head -6 || true

D=$(mktemp -d /tmp/rtprobe.XXXXXX)
cd "$D"
go tool pack x "$A" 2>&1 | head -3 || true
echo "--- unpacked:"
ls | head -8 || true

O=$(grep -l "$SYM" -- *.o 2>/dev/null | head -1 || true)
echo "--- object holding $SYM: $O"
if [ -n "$O" ]; then
    echo "--- go tool nm | grep:"
    go tool nm "$O" 2>&1 | grep "$SYM" | head -4 || true
    echo "--- go tool objdump -S -s $SYM (first 18 lines):"
    go tool objdump -S -s "$SYM" "$O" 2>&1 | head -18 || true
fi
cd /; rm -rf "$D"