#!/bin/bash

k=$1 
n=$2 # 64
g=$3 # 2
targets=$4
tag=$5
shift 5
options="$@"

embed=cgcnn92
epochs=500

mu="mixtures"
nu=${targets}

tasks="regression"
losses="L1"

leaf=${targets}_${k}-${n}-${g}
tag=${leaf}

mkdir results/${nu}
mkdir results/${nu}/$tag
echo $tag;

echo $k >> results/${nu}/${tag}/${tag}_log.txt
echo $n >> results/${nu}/${tag}/${tag}_log.txt
echo $h >> results/${nu}/${tag}/${tag}_log.txt
echo $d >> results/${nu}/${tag}/${tag}_log.txt
echo $g >> results/${nu}/${tag}/${tag}_log.txt
echo $r >> results/${nu}/${tag}/${tag}_log.txt

python examples/mixtures-example.py \
--train \
--evaluate \
--data-seed 0 \
--data-path datasets/${mu}/train_fold_${k}.csv \
--val-path datasets/${mu}/test_fold_${k}.csv \
--test-path datasets/${mu}/test_fold_${k}.csv \
--targets ${targets} \
--tasks ${tasks} \
--losses ${losses} \
--robust \
--epoch $epochs \
--elem-fea-len $n \
--n-graph $g \
--elem-emb $embed \
--model-name $tag \
 | tee -a -i results/${nu}/${tag}/${tag}_log.txt;
 cp models/${tag}/* results/${nu}/${tag}/. --backup=numbered
 rm -r models/${tag}
 mv results/results_${tag}.csv results/${nu}/${tag}/${tag}.csv
