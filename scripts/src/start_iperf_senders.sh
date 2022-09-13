#!/bin/bash

for (( i = 1; i <= $1; i=i+2 )) 
do
	port=$((5200+$i))
    iperf3 -V -f m -t 20 -c $3 -p $port > /dev/null &
    port=$((5200+$i+1))
    iperf3 -V -f m -t 20 -c $3 -p $port -C reno > /dev/null &
done

iperf3 -c $3 -p 5002 -u -b ${2}M -t 20  > /dev/null &



