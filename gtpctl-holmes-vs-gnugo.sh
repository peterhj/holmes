#!/bin/sh
set -e
set -u
GNUGO_FLAGS=
#GNUGO_FLAGS=--positional-superko
RUST_BACKTRACE=1 ./target/release/gtpctl &
sleep 1
../bin/gnugo ${GNUGO_FLAGS} --mode gtp --gtp-connect 127.0.0.1:6061 &
RUST_BACKTRACE=1 /data0/bin/operf ./target/release/holmes-gtp -h 127.0.0.1 -p 6060
