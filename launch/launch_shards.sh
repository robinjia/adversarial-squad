#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 <bundle_prefix> <num_shards> <script.sh>" 1>&2
  exit 1
fi
bundle_prefix="$1"
num_shards="$2"
script="$3"

for i in $(seq 0 $(expr $num_shards - 1))
do
  ./${script} ${bundle_prefix}-shard${i}.json
done
