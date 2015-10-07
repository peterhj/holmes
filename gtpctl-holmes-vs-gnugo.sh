#!/bin/sh
set -e
set -u
GNUGO_FLAGS=
#GNUGO_FLAGS=--positional-superko
RUST_BACKTRACE=1 ./target/release/gtpctl &
sleep 1
RUST_BACKTRACE=1 ./target/release/holmes-gtp -h 127.0.0.1 -p 6060 &
../bin/gnugo ${GNUGO_FLAGS} --mode gtp --gtp-connect 127.0.0.1:6061
