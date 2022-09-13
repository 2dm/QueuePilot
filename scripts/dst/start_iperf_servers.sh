#!/bin/bash

for i in $( eval echo {1..$1} ) 
do
	port=$((5200+$i))
	iperf3 -s -p $port > /dev/null &
done

iperf3 -s -p 5002 > /dev/null &

