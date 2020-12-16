SHELL := /bin/bash

CC = gcc -O3
RM = rm -f

EXECUTABLES = v0

all: $(EXECUTABLES)

v0: v0.c
	$(CC) $< -o $@ -lopenblas -lpthread -lm

clean:
	$(RM) *.o *~ $(EXECUTABLES)

default:
	all