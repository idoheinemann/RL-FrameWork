#!/bin/sh
while true;
do
	python3 -m example.fiar.fiar_main
	[ $? -eq 0 ] || break
done
