#include <opencv2/opencv.hpp>

int main() {
    // Load an image
    cv::Mat image = cv::imread("monkey.jpg");
    
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Display the image
    cv::imshow("Loaded Image", image);
    cv::waitKey(0); // Wait for any key to close the window

    return 0;
}