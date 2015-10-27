#!/bin/sh
set -e
set -u

#GNUGO_FLAGS="--level 8"
GNUGO_FLAGS="--level 1 --positional-superko"
GNUGO="../bin/gnugo ${GNUGO_FLAGS} --mode gtp --gtp-connect 127.0.0.1:6061"

HOLMES_PRELUDE=
#HOLMES_PRELUDE=/data0/bin/operf
#HOLMES_PRELUDE=/usr/local/cuda/bin/cuda-memcheck
HOLMES="./target/release/holmes-gtp -h 127.0.0.1 -p 6060"

RUST_BACKTRACE=1 ./target/release/gtpctl &
sleep 1
${GNUGO} &
RUST_BACKTRACE=1 ${HOLMES_PRELUDE} ${HOLMES}
