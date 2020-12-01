#!/bin/bash -e

set -e

pushd ci >/dev/null
./format.sh -d
popd >/dev/null
