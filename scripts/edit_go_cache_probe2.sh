#!/bin/sh
# Red-team probe (step-9.5 B2, follow-up): what exactly in a derived Go image's warm build cache
# names the repository's own module, and is any of the repository's compiled code there?
# Usage: docker run --rm -i --network none --user <uid> --entrypoint /bin/sh <image> -s < this
# $1 = module path (github.com/org/name), $2 = a repo-specific identifier to grep for.
set -eu

MOD="${1:-github.com/Masterminds/squirrel}"
NAME="${2:-squirrel}"

echo "--- -d files naming the module: size, first bytes, match context"
for d in $(grep -rl -a -m1 "$MOD" /opt/rlm-cache/go-build 2>/dev/null | head -4); do
    echo "== $d ($(stat -c %s "$d") bytes)"
    head -c 120 "$d" | tr -c '[:print:]\n' '.' ; echo
    grep -a -o -m3 ".\{0,40\}$MOD.\{0,40\}" "$d" | head -3 || true
done

echo "--- any cache file holding the repo name as a symbol ($NAME):"
hits=0
for f in $(find /opt/rlm-cache/go-build -type f 2>/dev/null); do
    if grep -q -a -m1 "$NAME" "$f" 2>/dev/null; then
        echo "hit $f ($(stat -c %s "$f") bytes)"
        hits=$((hits + 1))
    fi
    [ "$hits" -ge 8 ] && break
done
echo "name-hits: $hits"

echo "--- cache totals: $(find /opt/rlm-cache/go-build -type f | wc -l) files"

echo "--- module cache: any source of the module itself? (GOMODCACHE)"
GC=$(go env GOMODCACHE 2>/dev/null || echo /go/pkg/mod)
echo "gomodcache: $GC"
ls "$GC" 2>/dev/null | head -5 || true
find "$GC" -maxdepth 3 -name "*$NAME*" 2>/dev/null | head -5 || true