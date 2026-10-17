#!/bin/bash
# build.sh <a|b> [STEPS SYNTH_AT BS ACCUM] -> <x>/kernel.py
R=$(cd "$(dirname "$0")/.." && pwd); K=$R/gpu; x=$1
{ echo "TESTSET_B64 = '$(base64 -w0 $R/data/testset.tsv)'"; echo "BNNORM_B64 = '$(base64 -w0 $R/bntts/bnnorm.py)'"; cat $K/common.py $K/$x/main_tail.py; } \
 | sed -e "s/__STEPS__/${2:-300}/" -e "s/__SYNTH_AT__/${3:-200,300}/" -e "s/__BS__/${4:-2}/" -e "s/__ACCUM__/${5:-8}/" > $K/$x/kernel.py
echo "$K/$x/kernel.py"
