#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // Initialize webcam
    cv::VideoCapture cap(0, cv::CAP_DSHOW);  // Force DirectShow backend
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera!" << std::endl;
        return -1;
    }

    // Load face detector
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("assets/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load Haar Cascade XML!" << std::endl;
        return -1;
    }

     // Load ONNX emotion recognition model
    cv::dnn::Net emotionNet = cv::dnn::readNetFromONNX("assets/emotion-ferplus-8.onnx");

    // Emotion labels
    std::vector<std::string> emotions = {
        "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt"
    };

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect faces
        face_cascade.detectMultiScale(gray, faces);

        for (const auto& face : faces) {
            // Crop face region
            cv::Mat faceROI = gray(face);
            cv::Mat resized;
            cv::resize(faceROI, resized, cv::Size(64, 64));  // model input size

            // Convert to float and normalize
            resized.convertTo(resized, CV_32F, 1.0 / 255.0);

            // Create blob for the model: [1, 1, 64, 64]
            cv::Mat inputBlob = cv::dnn::blobFromImage(resized, 1.0, cv::Size(64, 64),
                                                      cv::Scalar(0), false, false, CV_32F);

            // Feed forward
            emotionNet.setInput(inputBlob);
            cv::Mat output = emotionNet.forward();

            // Find the max probability
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(output.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
            int labelId = classIdPoint.x;

            // Draw face rectangle and emotion label
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, emotions[labelId], 
                        cv::Point(face.x, face.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, 
                        cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Emotion Detector", frame);

        if (cv::waitKey(1) == 27) break;  // ESC to quit
    }

    return 0;
}
