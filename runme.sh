#!/bin/bash

./FIRESTARTER -t 30 -r 2>&1 \
    | grep QQQ \
    | tr -d "Q" \
    | tr -s " " \
    | sed -e "s/^ //" \
    > data.R

