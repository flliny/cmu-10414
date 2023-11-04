# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fpic `python3 -m pybind11 --includes`
LDFLAGS := -undefined dynamic_lookup
INCLUDE_DIR := /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers

SRC := src/simple_ml_ext.cpp
TARGET := src/simple_ml_ext.so

default:
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(SRC) -o $(TARGET) $(LDFLAGS)
