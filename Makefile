# Makefile for PhiInt c++ application.
# Nathan Frank, 05 June 2017.

CC = g++
CPPFLAGS = -Wall # -I/usr/local/include
LDFLAGS = -lcminpack  # -L/usr/local/lib -lgsl
OBJS = phi_int.o r0_int.o grba_int.o main.o
PHIPATH = ./phi_int/
R0PATH = ./r0_int/

ifeq ($(BUILD), profile)
CPPFLAGS += -pg -O0
LDFLAGS += -pg
PROG = GrbaInt
else ifeq ($(BUILD), shared)
CPPFLAGS += -O3
LDFLAGS += -shared -fPIC
PROG = libgrba_int.so
else
CPPFLAGS += -O3
PROG = GrbaInt
endif

all : $(PROG)

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS)

phi_int.o : $(PHIPATH)phi_int.h
	$(CC) $(CPPFLAGS) -c $(PHIPATH)phi_int.cpp

r0_int.o : $(R0PATH)r0_int.h
	$(CC) $(CPPFLAGS) -c $(R0PATH)r0_int.cpp

grba_int.o : grba_int.h  # <gsl/gsl_integration.h>
	$(CC) $(CPPFLAGS) -c grba_int.cpp

main.o : $(PHIPATH)phi_int.h $(R0PATH)r0_int.h grba_int.h
	$(CC) $(CPPFLAGS) -c main.cpp

clean:
	rm -f $(OBJS) GrbaInt libgrba_int.so

profile:
	make "BUILD=profile"

shared:
	make "BUILD=shared"
