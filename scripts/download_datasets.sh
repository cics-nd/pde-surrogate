#!/bin/bash

RES=$1

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}


case ${RES} in
	'64')
		GDRIVE_ID=1UVY767XNw9tO5fBzbYwo2wuQxDmk4JRo
		TAR_FILE=./datasets/64x64.tar
		;;
	'32')
		GDRIVE_ID=1ERud8ZBBRItTAh8co7GVZKRPCcoFxoHI
		TAR_FILE=./datasets/32x32.tar
		;;
esac

mkdir -p ./datasets

gdrive_download $GDRIVE_ID $TAR_FILE
tar -xvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
