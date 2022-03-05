#include<opencv2/opencv.hpp>
#include<iostream>
#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


// Namespace nullifies the use of cv::function();


#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

cv::Mat Inference(cv::Mat img, cv::Mat resultado, char argv[]) {

    //cv::Mat resultado = cv::Mat::zeros(256, 256, CV_32FC1);
   
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(argv);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Intrepter and does various set up
    // tasks so that the Interpreter can read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    //tflite::PrintInterpreterState(interpreter.get());

    // Fill input buffers
    // TODO(user): Insert code to fill input tensors.
    // Note: The buffer of the input tensor with index `i` of type T can
    // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

    float* input = interpreter->typed_input_tensor<float>(0);

    memcpy(input, img.data, sizeof(float) * img.rows * img.cols * img.channels());

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    //printf("\n\n=== Post-invoke Interpreter State ===\n");
    //tflite::PrintInterpreterState(interpreter.get());

    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

    float* output = interpreter->typed_output_tensor<float>(0);

    

    memcpy(resultado.data, output,
        sizeof(float) * resultado.rows * resultado.cols * resultado.channels());

    return resultado;

}


std::pair<cv::Mat, cv::Mat> CalcUVDisparityMap(cv::Mat disp_map)
{
    int l, c, pixel;  // local variables for row, line and pixel
    int intensity;	  // pixel intensity 

    //int* ptr1; // row pointer for 8 bits image 
    //unsigned short* ptr2; // row pointer for 16 bits image

    // U disparity map iamge
    cv::Mat u_disp = cv::Mat::zeros(cv::Size(disp_map.cols, 256), CV_8UC1);

    // V disparity map image
    cv::Mat v_disp = cv::Mat::zeros(cv::Size(256, disp_map.rows), CV_8UC1);

    // run accross the image and add 1 to the respective U/V disparity column
    for (l = 0; l < disp_map.rows; ++l)
    {
        //ptr1 = disp_map.ptr<int>(l);

        for (c = 0; c < disp_map.cols; ++c)
        {
            pixel = l * disp_map.cols + c;
            intensity = disp_map.data[pixel];

            if ((intensity > 0) && (intensity < 255))
            {
                pixel = intensity * u_disp.cols + c;
                u_disp.data[pixel] = u_disp.data[pixel] + 1;

                pixel = l * v_disp.cols + intensity;
                v_disp.data[pixel] = v_disp.data[pixel] + 1;
            }
        }
    }

    return std::make_pair(u_disp, v_disp);
}

std::pair<cv::Mat, cv::Mat> MaskSurface3(cv::Mat disp_map, cv::Mat v_disp, cv::Mat u_disp)
{
    // Free space mask
    cv::Mat mask1;

    // Obstacle Mask
    cv::Mat mask2;

    // Fill the destiny images with background value
    mask1 = cv::Mat::zeros(cv::Size(disp_map.cols, disp_map.rows), CV_8UC1);
    mask2 = cv::Mat::zeros(cv::Size(disp_map.cols, disp_map.rows), CV_8UC1);

    int l, c, pixel, v_disp_pixel, u_disp_pixel;	// local variables for row, column and pixel
    int intensity;		// pixel intensity
    int intensity_B_v;	//pixel intensity
    int intensity_B_u;//pixel intensity


    // run the images associating the red pixels in the v-disparities in the mask1 e mask2
    for (l = 0; l < disp_map.rows; ++l)
    {


        for (c = 0; c < disp_map.cols; ++c)
        {
            pixel = l * disp_map.cols + c;

            intensity = disp_map.data[pixel];

            //--------- Marca os obstaculos ----------------------------					

            v_disp_pixel = l * v_disp.cols + intensity;
            u_disp_pixel = intensity * u_disp.cols + c;
            intensity_B_v = v_disp.data[v_disp_pixel];
            intensity_B_u = u_disp.data[u_disp_pixel];

            //cout << ("Intensidade de V �:")<<(int)intensity_B_v<<(" e a intenside de U � ")<<(int)intensity_B_u << endl;

            if ((intensity_B_v > 25) && (intensity_B_u > 25))
            {
                pixel = l * disp_map.cols + c;
                mask2.data[pixel] = 1;
            }
            //----------------------------------------------------------
        }
    }


    // Espande as regioes brancas da imagem
//cv::dilate(mask1, mask1, cv::Mat(), cv::Point(-1, -1), 2);
    cv::dilate(mask2, mask2, cv::Mat(), cv::Point(-1, -1), 2);


    return std::make_pair(mask1, mask2);
}


int main()
{
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

//================================Read Image==========================================
    cv::Mat img = cv::imread("input/4,93.jpg");

    resize(img, img, cv::Size(256, 256), 0, 0, cv::INTER_AREA);

    cv::Mat img2 = img;

//================================Convert to Float==========================================

    img.convertTo(img, CV_32FC3);

    //std::cout << img.type() << std::endl;

    //cv::imshow("Output", img);
    //cv::waitKey(0);
    // 
//================================RUN MIDAS==========================================
    char modelfile[] = "model_MIDAS.tflite";

    cv::Mat resultado = cv::Mat::zeros(256, 256, CV_32FC1);

    cv::Mat saida = Inference(img, resultado, modelfile);
    saida = saida / 16;
    cv::Mat resultado2 = cv::Mat::zeros(256, 256, CV_8UC1);
    saida.convertTo(resultado2, CV_8UC1);

    /*
    minMaxLoc(resultado2, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    cv::imshow("output", resultado2);
    cv::waitKey(0);
   
    */

    cv::Mat disp_map = resultado2;

//================================TEST RESULT MIDAS==========================================
    //cv::resize(resultado2, resultado2, cv::Size(1280, 720), cv::INTER_LINEAR);



    minMaxLoc(resultado2, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    //cv::imshow("Output2", resultado2);
    //cv::waitKey(0);

//================================CONVERT TO U/V MAP==========================================

    std::pair<cv::Mat, cv::Mat> uv_disp_map = CalcUVDisparityMap(resultado2);
    //cv::imshow("U-map", uv_disp_map.first);
    //cv::imshow("V-map", uv_disp_map.second);
    //cv::waitKey(0);

//================================PREPARE IMAGES TO MODEL==========================================

    cv::Mat input;

    vconcat(uv_disp_map.second, uv_disp_map.first, input);

    cv::normalize(input, input, 0, 255, cv::NORM_MINMAX, -1, cv::noArray());

    input.convertTo(input, CV_32FC1);

    input = (input / 127.5) - 1;

    

    minMaxLoc(input, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    //cv::imshow("output", input);
    //cv::waitKey(0);

/*
    resultado = cv::Mat::zeros(512, 256, CV_32FC1);

    std::cout << resultado.cols << std::endl;
    std::cout << resultado.rows << std::endl;
    std::cout << resultado.channels() << std::endl;


    std::cout << input.cols << std::endl;
    std::cout << input.rows << std::endl;
    std::cout << input.channels() << std::endl;

    cv::imshow("Input", input);
    cv::waitKey(0);
*/
//================================RUN DETECT U/V MODEL==========================================

    char modelUVfile[] = "modelUV.tflite";

    resultado = cv::Mat::zeros(512, 256, CV_32FC1);

    saida = Inference(input, resultado, modelUVfile);
    //resultado2 = cv::Mat::zeros(512, 256, CV_8UC1);
    //saida.convertTo(resultado2, CV_8UC1);

    minMaxLoc(saida, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    //cv::imshow("output", saida);
    //cv::waitKey(0);
//================================PREPARE TO MASK==========================================

    saida = saida + (minVal*(-1));
    saida = saida * 255;
    //resultado2 = cv::Mat::zeros(512, 256, CV_8UC1);
    saida.convertTo(saida, CV_8UC1);

    minMaxLoc(saida, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    //cv::imshow("output", saida);
    //cv::waitKey(0);


//================================RUN MASK==========================================


    cv::Mat v_disp = saida(cv::Range(0, 256), cv::Range(0, 256));
    cv::Mat u_disp = saida(cv::Range(256, 512), cv::Range(0, 256));
/*
    minMaxLoc(v_disp, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    minMaxLoc(u_disp, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "min val: " << minVal << std::endl;
    std::cout << "max val: " << maxVal << std::endl;

    cv::imshow("output", v_disp);
    cv::imshow("output2", u_disp);
    cv::waitKey(0);

*/
    std::pair<cv::Mat, cv::Mat> mask = MaskSurface3(disp_map, v_disp, u_disp);


//================================PRINT MASK==========================================

    std::vector<cv::Mat> channels(3);
    cv::split(img2, channels);

    cv::Mat masks;
    masks = 1 - mask.second;

    channels[0] = masks.mul(channels[0]); // Activate the red color as obstacles

    cv::merge(channels, img2);

    resize(img2, img2, cv::Size(640, 480), cv::INTER_LINEAR);

 
    //imshow("u disp_map", uv_disp_map.first);
    //imshow("v disp_map", uv_disp_map.second);
    //imshow("Detect Obeject U disp_map", uv_disp_detect.second);
    //imshow("Detect Obeject V disp_map", uv_disp_detect.first);
    imshow("Mask", mask.second * 255);
    imshow("Image", img2);
    cv::waitKey(0);

    return 0;
}