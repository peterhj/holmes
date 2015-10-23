#!/bin/sh
set -e
set -u
GNUGO="../bin/gnugo --mode gtp --gtp-connect 127.0.0.1:6060"
FUEGO="../bin/fuego --config conf/fuego_10k.conf --quiet"
#FUEGO="../bin/fuego --quiet --opt-num-playouts 3072 --opt-num-threads 1"
./target/release/gtpctl &
sleep 1
${GNUGO} &
nc -c "${FUEGO}" 127.0.0.1 6061
