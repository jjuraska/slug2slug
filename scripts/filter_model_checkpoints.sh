#!/usr/local/bin/bash

cd ~/slug2slug/model/
rm model.ckpt-{???,????}1.*
sed -i '/.*01"$/d' ./checkpoint
