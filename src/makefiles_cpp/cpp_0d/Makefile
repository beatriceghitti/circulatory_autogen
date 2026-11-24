# Compiler
CXX = g++
#CXX = mpic++
# Compiler flags (add -g for debug, or -O2/-O3 for optimization, or -Wall for enabling warning messages)
CXXFLAGS = -Wall -O2 -std=c++17 $(INC_DIR)

# SUNDIALS paths from Spack
# Serial version of SUNDIALS - both lines pick the same serial build, if no sundials+mpi is installed.
SUNDIALS_PREFIX := $(shell spack location -i sundials)
# SUNDIALS_PREFIX := $(shell spack location -i sundials~mpi)
INC_DIR = -I$(SUNDIALS_PREFIX)/include
LIB_DIR = -L$(SUNDIALS_PREFIX)/lib

# Linker flags
LDFLAGS  = $(LIB_DIR)
# Linker libraries
LDLIBS = -lsundials_cvodes \
	 -lsundials_sunmatrixdense \
         -lsundials_sunlinsoldense \
         -lsundials_nvecserial \
         -lsundials_core \
         -lm

# Source files
SRCS = model0d.cc
# Object files (auto-generated from SRCS)
OBJS := $(SRCS:.cpp=.o) # OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)
# Output/target binary
TARGET = model0d

# Build rules
.PHONY: all clean

all: $(TARGET)

# Link target	
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Header dependencies: when compiling, the compiler also writes a .d file next to each object file (-MMD), listing
# the headers it includes (the model header), so that make recompiles the object file when one of
# them changes, e.g. after regenerating the model, without 'make clean'. -MP: a header that no longer exists does not
# stop make. An object file without its .d file (e.g. built before) is recompiled once.
DEPFLAGS = -MMD -MP
DEPS := $(OBJS:.o=.d)

# Compile source files
%.o: %.cpp %.d
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@
%.o: %.cc %.d
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# .d files: written by the compiler (no rule to make them), and read only if they exist
$(DEPS):
include $(wildcard $(DEPS))

# Clean
clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)
