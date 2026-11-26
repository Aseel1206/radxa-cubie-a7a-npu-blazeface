#!/bin/bash
# float TFLite reference vs Acuity simulator (float / pcq / i16) on one image
source /workspace/mp/scripts/env.sh
D=/workspace/mp/face_det; cd $D
sed 's#path: [^ ]*#path: /workspace/mp/test128/dataset.txt#' net_inputmeta.yml > net_inputmeta_test.yml
for t in pcq i16; do
  $PEG inference --model net.json --model-data net.data --dtype quantized --model-quantize net_$t.quantize --with-input-meta net_inputmeta_test.yml --device CPU --output-dir inf_$t --iterations 1 > inf_$t.log 2>&1
done
python3 /workspace/mp/scripts/sim_compare.py
chmod -R a+rwX $D
