#########################################################################
# File Name:    get_split.sh
# Author:       Ruibin Liu
# mail:         ruibinliuphd@gmail.com
# Created Time: Mon 27 May 2024 01:52:54 PM EDT
#########################################################################
#!/bin/bash
split=$1
prefix='ligcys'
rm -r ${prefix}_$split/raw/LigCys/data
rm -r ${prefix}_$split/splits/split-by-clustering/data
nohup python prepare_lmdb.py -nt 4 -p ../../../data/data_splitting/${prefix}_$split/${prefix}_pos.csv -n ../../../data/data_splittings/${prefix}_$split/${prefix}_neg.csv -tr ../../../data/data_splittings/${prefix}_$split/train_ids.txt -v ../../../data/data_splittings/${prefix}_$split/val_ids.txt -t ../../../data/data_splittings/${prefix}_$split/test_ids.txt -cv 10 -s ../../../data/data_splittings/atom3d_pdbs ${prefix}_$split > ${prefix}_$split.log &

