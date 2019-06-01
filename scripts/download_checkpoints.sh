#!/bin/bash

KLE=$1
NUM_TRAIN=$2

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# kle100ng32n4096beta150.tar
GDRIVE_ID=1--1F6AvzyEKyta3yfdtdpfGR9sPebbd6

TARGET_DIR=./experiments/cglow/reverse_kld
mkdir -p $TARGET_DIR

TAR_FILE=./experiments/cglow/reverse_kld/kle100ng32n4096beta150.tar

gdrive_download $GDRIVE_ID $TAR_FILE
tar -xvf $TAR_FILE -C $TARGET_DIR

rm $TAR_FILE
