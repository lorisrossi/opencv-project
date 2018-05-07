CPP=g++
CPPFLAGS= -std=c++11 -Wall -Wextra -fsanitize=address

LIBS=-lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio

hello_world: main.cpp
	$(CPP) -o $@ $^ $(CPPFLAGS) $(LIBS)

clean:
	rm -f hello_world

.PHONY: clean
