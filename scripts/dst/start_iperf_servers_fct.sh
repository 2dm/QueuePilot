#!/bin/bash

for i in $( eval echo {1..$1} )
do
	port=$((5200+$i))
	iperf3 -s -p $port > /dev/null &
done

iperf3 -s -p 5550 > /dev/null &

for i in $( eval echo {1..50} ) 
do
	cport=$((4999+$i))
	port=$((5050+$i))
	nuttcp -S -p$port -P$cport --idle-data-timeout 100/200/300 &
done


