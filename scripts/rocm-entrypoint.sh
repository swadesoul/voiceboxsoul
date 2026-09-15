#!/bin/sh
set -e

# Prepare persistent bind-mounted directories before dropping privileges.
# Hostinger/VPS creates these host paths as root, while the application runs
# as the non-root `voicebox` user. Without this, SQLite cannot create
# /app/data/voicebox.db and startup fails with "unable to open database file".
mkdir -p \
    /app/data \
    /app/data/generations \
    /app/data/profiles \
    /app/data/cache \
    /home/voicebox/.cache/huggingface

chown voicebox:voicebox \
    /app/data \
    /app/data/generations \
    /app/data/profiles \
    /app/data/cache \
    /home/voicebox/.cache \
    /home/voicebox/.cache/huggingface

# Join whatever groups own GPU device nodes, then drop to the app user.
# AMD/ROCm nodes are handled when present; NVIDIA access is normally supplied
# by Docker's GPU runtime and does not require these device groups.
for dev in /dev/kfd /dev/dri/render*; do
    [ -e "$dev" ] || continue
    gid=$(stat -c %g "$dev")
    grp=$(getent group "$gid" | cut -d: -f1)
    [ -n "$grp" ] || {
        grp="gpu$gid"
        groupadd -g "$gid" "$grp"
    }
    usermod -aG "$grp" voicebox
done

exec gosu voicebox "$@"
