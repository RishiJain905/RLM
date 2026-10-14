#!/bin/sh
# Red-team probe (documentation/step-9.5-red-team.md, B2): does a derived Go image's warm build
# cache hold a compiled archive of the repository's own module — built from the intact, pre-break
# (gold) source? Run it inside one throwaway container:
#
#   docker run --rm -i --network none --user <uid> --entrypoint /bin/sh rlm-edit-<repo>:<hash> -s \
#       < scripts/edit_go_cache_probe.sh
#
# Pass the module path as $1 (default: github.com/Masterminds/squirrel). Read-only: it greps the
# cache, lists `go tool nm` symbols and disassembles one with `go tool objdump`. Nothing is
# written inside the container.
set -eu

MOD="${1:-github.com/Masterminds/squirrel}"

echo "--- -d metadata entries naming the module"
grep -rl -a -m1 "$MOD" /opt/rlm-cache/go-build 2>/dev/null | head -10 || true
echo "d-hits: $(grep -rl -a -m1 "$MOD" /opt/rlm-cache/go-build 2>/dev/null | head -40 | wc -l)"

echo "--- -a archives naming the module"
n=0
first=""
for a in $(find /opt/rlm-cache/go-build -name '*-a' 2>/dev/null); do
    if grep -q -a -m1 "$MOD" "$a" 2>/dev/null; then
        echo "hit $a"
        [ -z "$first" ] && first="$a"
        n=$((n + 1))
    fi
    [ "$n" -ge 5 ] && break
done
echo "a-hits (of first pass): $n"

echo "--- archive: $first"
echo "--- go tool nm, module symbols:"
go tool nm "$first" 2>&1 | grep -i "$(basename "$MOD")" | head -10 || true

SYM=$(go tool nm "$first" 2>/dev/null | grep -i "$(basename "$MOD")" | head -1 | awk '{print $NF}')
echo "--- go tool objdump -s $SYM (first 14 lines):"
go tool objdump -s "$SYM" "$first" 2>&1 | head -14 || true