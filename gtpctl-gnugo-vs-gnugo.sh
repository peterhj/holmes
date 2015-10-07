#!/bin/sh
./target/release/gtpctl &
sleep 1
../bin/gnugo --mode gtp --gtp-connect 127.0.0.1:6060 &
../bin/gnugo --mode gtp --gtp-connect 127.0.0.1:6061
