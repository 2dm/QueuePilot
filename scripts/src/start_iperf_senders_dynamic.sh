#!/bin/bash

for i in $( eval echo {1..100} ) 
do
	port=$((5200+$i))
	delay=5
	(sleep $delay; iperf3 -V -f m -t 30 -c 192.168.5.1 -p $port > /dev/null) &
done

for i in $( eval echo {101..150} )
do
	port=$((5200+$i))
	delay=5
	(sleep $delay; iperf3 -V -f m -t 20 -c 192.168.5.1 -p $port > /dev/null) &
done

for i in $( eval echo {151..200} )
do
	port=$((5200+$i))
	delay=5
	(sleep $delay; iperf3 -V -f m -t 30 -c 192.168.5.1 -p $port > /dev/null) &
done

for i in $( eval echo {201..300} ) 
do
	port=$((5200+$i))
	delay=15
	(sleep $delay; iperf3 -V -f m -t 10 -c 192.168.5.1 -p $port > /dev/null) &
done

iperf3 -c 192.168.5.1 -p 5002 -u -b 80M -t 40  > /dev/null &
