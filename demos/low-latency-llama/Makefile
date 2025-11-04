# Compiler
HIPCC?=hipcc

# Default GPU target (e.g., MI300X)
ifndef GPU
GPU=MI300X
endif

TARGET=mk_llama
SRC=llama.cu

# Python version configuration
ifndef PYTHON_VERSION
PYTHON_VERSION=3.13
endif

# Common flags for the host and device compiler
COMMON_FLAGS=-DNDEBUG -O3 -std=c++20 -fPIC
# Host compiler specific flags
HOST_FLAGS=-Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
# Linker flags
LINK_FLAGS=-lrt -lpthread -ldl -lhipsparse

# Base HIPCC flags
HIPCCFLAGS=$(COMMON_FLAGS) $(HOST_FLAGS) $(LINK_FLAGS) -shared
# Include paths and Python flags
HIPCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -lpython${PYTHON_VERSION}

# Conditional setup based on the target AMD GPU
ifeq ($(GPU),MI200)
# gfx90a: MI200 series
HIPCCFLAGS+= --offload-arch=gfx90a -DKITTENS_MI200
else ifeq ($(GPU),MI250)
# gfx90a: MI200 series
HIPCCFLAGS+= --offload-arch=gfx90a -DKITTENS_MI250
else ifeq ($(GPU),MI300X)
# gfx942: MI300X series
HIPCCFLAGS+= --offload-arch=gfx942 -DKITTENS_MI300X
else
# Default to MI300X architecture
HIPCCFLAGS+= --offload-arch=gfx942 -DKITTENS_MI300X
endif

# Default target
all: $(TARGET)

# run: $(TARGET)
# 	python test.py

$(TARGET): $(SRC)
	$(HIPCC) $(SRC) $(HIPCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)