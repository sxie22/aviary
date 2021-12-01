#!/bin/bash

k=$1 
n=$2 # 64
h=$3 # 64
g=$4 # 2
d=$5 # 3
targets=$6  # "BV_3D V_ox V_red E_f"

r=5 
embed=cgcnn92
epochs=5

mu="indiv_13245"
nu=${targets}

tasks="regression"
losses="L1"

leaf=${targets}_${k}-${n}-${h}-${g}-${d}
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

python globals-example.py \
--train \
--evaluate \
--robust \
--data-path datasets/${mu}/train_fold_${k}.csv \
--val-path datasets/${mu}/test_fold_${k}.csv \
--test-path datasets/${mu}/test_fold_${k}.csv \
--globals-path ./global_22.dat \
--targets ${targets} \
--tasks ${tasks} \
--losses ${losses} \
--epoch $epochs \
--data-seed 0 \
--elem-fea-len $n \
--n-graph $g \
--h-fea-len $h \
--radius $r \
--n-hidden $d \
--elem-emb $embed \
--model-name $tag \
 | tee -a -i results/${nu}/${tag}/${tag}_log.txt;
 cp models/${tag}/* results/${nu}/${tag}/. --backup=numbered
 rm -r models/${tag}
 mv results/results_${tag}.csv results/${nu}/${tag}/${tag}.csv
