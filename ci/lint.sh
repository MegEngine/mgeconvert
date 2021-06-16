#!/usr/bin/env bash
set -e
cd $(dirname $0)/..

echo "Running mypy..."
mypy mgeconvert test --show-error-codes || mypy_ret=$?
echo "Running pylint ..."
pylint mgeconvert test --rcfile=.pylintrc || pylint_ret=$?

if [ "$mypy_ret" ]; then
    exit $mypy_ret
fi

if [ "$pylint_ret" ]; then
    exit $pylint_ret
fi

echo "All lint steps passed!"
