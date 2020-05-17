#!/bin/sh
while true;
do
	pypy3 -m example.fiar.fiar_main
	[ $? -eq 0 ] || break
done
