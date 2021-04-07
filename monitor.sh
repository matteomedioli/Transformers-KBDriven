#!/bin/bash
if [ -z "$1" ]
then
N=10
else
N=$1
fi
while [ 1 -gt 0 ]
do
tail -n $N nohup.out
sleep 1
clear
done
