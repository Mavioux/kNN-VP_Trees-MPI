SHELL := /bin/bash

CC = gcc -O3
MPICC = mpicc
RM = rm -f

EXECUTABLES = v0 v1 v1_1 v2

all: $(EXECUTABLES)

v0: v0.c
	$(CC) $< -o ../out/$@ -lopenblas -lpthread -lm

v1: v1.c
	$(MPICC) $< -o ../out/$@ -lopenblas -lpthread -lm
	
v1_1: v1_1.c
	$(MPICC) $< -o ../out/$@ -lopenblas -lpthread -lm

v2: v2.c
	$(MPICC) $< -o ../out/$@ -lopenblas -lpthread -lm

clean:
	$(RM) *.o *~ $(EXECUTABLES)

default:
	all