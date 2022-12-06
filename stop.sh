#!/bin/bash
ps aux | grep 'libcom_server' |grep -v grep| awk '{print $2}'| xargs kill -9
