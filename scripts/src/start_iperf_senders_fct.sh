#!/bin/bash

measures=50
flows=$1

for i in $( eval echo {1..$flows} ) 
do
	port=$((5201+$4+$i))
    iperf3 -i 0 -f m -t 60 -c $3 -p $port > /dev/null &
done
 
iperf3 -i 0 -c $3 -p 5550 -u -b ${2}M -t 60  > /dev/null &

sleep 5

for i in $( eval echo {1..$measures} ) 
do
  cport=$((4999+$i))
	port=$((5100+$i))
	sleep 0.01 ; nuttcp -l$7 -n1 -p$port -P$cport $3 &
done

