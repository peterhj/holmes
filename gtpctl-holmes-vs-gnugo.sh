#!/bin/sh
set -e
set -u

BLACK_PORT="$1"
WHITE_PORT="$2"
GNUGO_SEED="$3"

#GNUGO_FLAGS="--level 8"
#GNUGO_FLAGS="--level 0 --chinese-rules --positional-superko"
GNUGO_FLAGS="--level 10 --chinese-rules"
GNUGO="../bin/gnugo-3.8 ${GNUGO_FLAGS} --seed ${GNUGO_SEED} --mode gtp --gtp-connect 127.0.0.1:${WHITE_PORT}"

HOLMES_PRELUDE=
#HOLMES_PRELUDE=/data0/bin/operf
#HOLMES_PRELUDE=/usr/local/cuda/bin/cuda-memcheck

#HOLMES="./target/debug/holmes-gtp -h 127.0.0.1 -p 6060"
#HOLMES="./target/release/holmes-gtp -h 127.0.0.1 -p 6060"
#HOLMES="experiments/logs/holmes_3layer_uct1k_fix-vs-gnugo/holmes-gtp -h 127.0.0.1 -p ${BLACK_PORT}"

#HOLMES="./target/debug/holmes-gtp -h 127.0.0.1 -p ${BLACK_PORT}"
HOLMES="./target/release/holmes-gtp -h 127.0.0.1 -p ${BLACK_PORT}"

RUST_BACKTRACE=1 ./target/release/gtpctl ${BLACK_PORT} ${WHITE_PORT} &
sleep 1
${GNUGO} &
RUST_BACKTRACE=1 ${HOLMES_PRELUDE} ${HOLMES}
