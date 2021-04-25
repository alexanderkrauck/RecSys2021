#!/bin/bash

while IFS='' read -r LINE || [ -n "${LINE}" ]; do
    if echo "$LINE" | grep -q -E '.lzo.index'; then
      echo "skipping an index file"
      continue
    fi
    NAME="$(grep -o '[a-zA-Z0-9_-]*\.lzo' <<< "$LINE")"
    #echo "using line" #$LINE
    wget "$LINE" -O ./dl/"$NAME"
done < ./training_urls.txt
