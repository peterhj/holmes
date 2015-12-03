#!/bin/sh
set -e
set -u

BLACK_PORT="$1"
WHITE_PORT="$2"
REFEREE_PORT="$3"
GNUGO_SEED="$4"

GNUGO_FLAGS="--level 10 --chinese-rules"
GNUGO="../bin/gnugo-3.8 ${GNUGO_FLAGS} --seed ${GNUGO_SEED} --mode gtp --gtp-connect 127.0.0.1:${REFEREE_PORT}"

HOLMES="./target/release/holmes-gtp -h 127.0.0.1 -p ${BLACK_PORT}"

#PACHI_FLAGS="-e uct -r chinese -f data/book-fuego.dat -t =10000"
PACHI_FLAGS="-e uct -r chinese -t =10000 threads=1,pondering=0"
PACHI="../bin/pachi-11 ${PACHI_FLAGS} -g 127.0.0.1:${WHITE_PORT}"

RUST_BACKTRACE=1 ./target/release/gtp_ref_ctrl ${BLACK_PORT} ${WHITE_PORT} ${REFEREE_PORT} &
sleep 5
${GNUGO} &
${PACHI} &
RUST_BACKTRACE=1 ${HOLMES}
