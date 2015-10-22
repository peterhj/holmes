#!/bin/sh
GNUGO="../bin/gnugo --mode gtp --gtp-connect 127.0.0.1:6060"
FUEGO="../bin/fuego --quiet"
set -e
set -u
./target/release/gtpctl &
sleep 1
${GNUGO} &
nc -c "${FUEGO}" 127.0.0.1 6061
