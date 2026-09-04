#!/bin/sh
# Creates vendor.tar.xz from local monorepo crates + crates.io deps.
# Must be run from bindings/r/src/. Leaves vendor/ in place for cargo builds.
set -e

rm -rf vendor vendor.tar.xz
mkdir -p vendor

# Copy local monorepo crates (paths relative to bindings/r/src/)
cp -rL ../../../crates/fastLoess vendor/fastLoess
cp -rL ../../../crates/loess-rs vendor/loess-rs

# Strip build artefacts and noise
rm -rf vendor/fastLoess/target vendor/loess-rs/target
rm -f vendor/fastLoess/Cargo.lock vendor/loess-rs/Cargo.lock
for d in tests benches examples doc docs assets .github .config; do
	rm -rf "vendor/fastLoess/$d" "vendor/loess-rs/$d" 2>/dev/null || true
done
rm -f vendor/fastLoess/README.md vendor/fastLoess/CHANGELOG.md \
	vendor/loess-rs/README.md vendor/loess-rs/CHANGELOG.md

# Patch vendored fastLoess: remove version pin on loess-rs path dep
sed -i.bak \
	-e 's|loess-rs = { path = "\.\./loess-rs", version = "[^"]*", |loess-rs = { path = "../loess-rs", |' \
	vendor/fastLoess/Cargo.toml
rm -f vendor/fastLoess/Cargo.toml.bak

# Checksum placeholders for manually-placed path-dep crates
printf '{"files":{},"package":null}' >vendor/loess-rs/.cargo-checksum.json
printf '{"files":{},"package":null}' >vendor/fastLoess/.cargo-checksum.json

# Temporarily isolate from the monorepo workspace so cargo vendor scopes
# only to this package, then restore the clean Cargo.toml immediately after.
# Restoring from a backup (rather than sed-deleting the appended lines)
# avoids leaving stray blank lines behind on every run.
cp Cargo.toml Cargo.toml.orig
printf '\n\n[patch.crates-io]\nloess-rs = { path = "vendor/loess-rs" }\n' >>Cargo.toml
cargo vendor -q --no-delete vendor
mv Cargo.toml.orig Cargo.toml

# Drop directories that bulk up the archive
for d in tests benches examples doc docs assets .github .config; do
	rm -rf "vendor/$d" vendor/*/"$d" 2>/dev/null || true
done
for f in vendor/*/Makefile; do [ -f "$f" ] && rm -f "$f"; done || true
rm -f vendor/*/CITATION.cff vendor/*/CITATION

# Nullify file-level checksums so removed test/bench/doc files don't cause
# "failed to open file" when cargo verifies vendor integrity
for f in vendor/*/.cargo-checksum.json; do
	[ -f "$f" ] || continue
	python3 -c "import json; p='$f'; d=json.load(open(p)); d['files']=[]; json.dump({'files':{},'package':d.get('package')},open(p,'w'))"
done

# Create reproducible archive; include Cargo.lock for reproducible installs
tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
	--xz --create --file=vendor.tar.xz --exclude='*/Makefile' vendor Cargo.lock \
	2>/dev/null ||
	tar --xz --create --file=vendor.tar.xz --exclude='*/Makefile' vendor Cargo.lock

echo "vendor.tar.xz created"
# vendor/ is left in place for subsequent cargo builds in the same session
