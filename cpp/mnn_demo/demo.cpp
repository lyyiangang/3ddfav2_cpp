//./mnn_demo ../../../models/face3d.mnn ../../../data/roi_face_120x120.png
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>
#include <string>
#include <limits>


#include "face3d_mnn.hpp"
using namespace  std;

int plot_circle(uint8_t* src, int point[2], int width, int height) {
    const int CIRCLE_RADIUS =1;
    for (int y = -CIRCLE_RADIUS; y < (CIRCLE_RADIUS + 1); ++y) {
        for (int x = -CIRCLE_RADIUS; x < (CIRCLE_RADIUS + 1); ++x) {
            const int xx = point[0] + x;
            const int yy = point[1] + y; 
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                int index  = yy * width + xx;
                src[index * 3] = 255;
                src[index * 3 + 1] = 255;
                src[index * 3 + 2] = 255;
            }
        }
    }
    return 0;
}

// convert to image coordinate system
void post_process(std::vector<float>& pts){
    float min_z = std::numeric_limits<float>::max();
    for(int ii = 0; ii < 68; ++ii){
        pts[ii * 3 ] -= 1.0;
        pts[ii * 3 + 2] -= 1.0;
        pts[ii * 3 + 1] = 120.0 - pts[ii * 3 + 1];
        if(pts[ii * 3 + 2] < min_z)
            min_z = pts[ii * 3 + 2];
    }
    for(int ii = 0; ii < 68; ++ii){
        pts[ii * 3 + 2] -= min_z;
    }
}

int main(int argc, char** argv){
    if(argc != 3){
        cout<<"wrong inputs. arguments like this demo.out model.mnn test.png";
        return 1;
    }
    string model_path = argv[1];
    string img_name = argv[2];
    int originalWidth;
    int originalHeight;
    int originChannel;
    auto inputImage = stbi_load(img_name.c_str(), &originalWidth, &originalHeight, &originChannel, 3);
    if (nullptr == inputImage) {
        cout<<"can not open "<<img_name<<"\n";
        return 1;
    }
    const auto rgbPtr = reinterpret_cast<uint8_t*>(inputImage);
    Face3d face_model(model_path);
    std::vector<float> lnds = face_model.Predict(rgbPtr);
    post_process(lnds);
    for(int ii = 0; ii < 68; ++ii){
        int pt[2] = {lnds[ii * 3], lnds[ii * 3 + 1]};
        plot_circle(rgbPtr, pt, originalWidth, originalHeight);
    }    
    std::string out_name = "mnn_det_result.png";
    stbi_write_png(out_name.c_str(), originalWidth, originalHeight, 3, inputImage, 3 * originalWidth);
    stbi_image_free(inputImage);
    std::cout<<"output detect result to "<< out_name<<std::endl;
    return 0;
}