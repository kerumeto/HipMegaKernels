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

# Common flags for host and device
# Note: -fPIC is strictly required for shared libraries
COMMON_FLAGS=-DNDEBUG -O3 -std=c++20 -fPIC

# Host compiler specific flags
# Removed -fPIE (only for executables) and strict-aliasing (safer for type-punning)
HOST_FLAGS=-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing

# Linker flags
# Removed explicit -lpython linkage (best practice for Linux extensions)
LINK_FLAGS=-lrt -lpthread -ldl -lhipsparse

# Base HIPCC flags
HIPCCFLAGS=$(COMMON_FLAGS) $(HOST_FLAGS) $(LINK_FLAGS) -shared

# Include paths
# Using --includes ensures we get Python headers without manually guessing paths
HIPCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes)

# Architecture Setup
# gfx90a = MI210, MI250, MI250X
# gfx942 = MI300A, MI300X
ifeq ($(GPU),MI200)
    HIPCCFLAGS+= --offload-arch=gfx90a -DKITTENS_MI200
else ifeq ($(GPU),MI250)
    HIPCCFLAGS+= --offload-arch=gfx90a -DKITTENS_MI250
else ifeq ($(GPU),MI300X)
    HIPCCFLAGS+= --offload-arch=gfx942 -DKITTENS_MI300X
else
    # Default fallback
    HIPCCFLAGS+= --offload-arch=gfx942 -DKITTENS_MI300X
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	# Ensure the line below starts with a TAB, not spaces
	$(HIPCC) $(SRC) $(HIPCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)