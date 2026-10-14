#!/bin/sh
# Red-team probe (step-9.5 B2, third pass): can the repository's compiled pre-break code in a
# derived Go image's warm build cache actually be *read*? Finds an archive naming the module,
# lists its symbols with `go tool nm`, and disassembles one with `go tool objdump`.
# Usage: docker run --rm -i --network none --user <uid> --entrypoint /bin/sh <image> -s < this
# $1 = module path, $2 = symbol substring to disassemble (default: the repo name).
set -eu

MOD="${1:-github.com/allegro/bigcache/v3}"
SYM="${2:-getNeededSize}"

found=""
for f in $(grep -rl -a -m1 "$MOD" /opt/rlm-cache/go-build 2>/dev/null | head -30); do
    case "$f" in
        *-d) ;;
        *) continue ;;
    esac
    if go tool nm "$f" >/dev/null 2>&1; then
        found="$f"
        break
    fi
done
echo "--- first nm-readable archive naming the module: $found"

echo "--- symbols matching $SYM:"
go tool nm "$found" 2>/dev/null | grep "$SYM" | head -5 || echo "(none)"

echo "--- objdump -s $SYM:"
go tool objdump -s "$SYM" "$found" 2>&1 | head -16 || true