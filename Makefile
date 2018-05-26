LDIR=lib
ODIR=build

CXX=g++
CXXFLAGS= -std=c++11 -Wall -Wextra -fsanitize=address -I$(LDIR)

LIBS=-lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio

_DEPS=io.hpp hog.hpp svm.hpp
DEPS=$(patsubst %,$(LDIR)/%,$(_DEPS))

_OBJ=main.o io.o hog.o svm.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(LDIR)/%.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR)/%.o: main.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

main: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/*.o

.PHONY: clean
