#!/usr/bin/env bash

wget https://drive.usercontent.google.com/download\?id\=1LpjC4pNCUH51U_QuEA-I1oY6dYjfb7AL\&confirm\=t \
-O train.tar.gz && \
tar -zxvf train.tar.gz && rm -rf train.tar.gz
