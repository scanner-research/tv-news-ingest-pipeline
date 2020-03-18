/*
 * File: detect_black_frames.cpp
 * -----------------------------
 * Detects black frames in a video and write the detected frame numbers to a
 * JSON file.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <getopt.h>
#include <opencv2/opencv.hpp>

using namespace std;

typedef vector<string> PathsList;
typedef vector<size_t> FramesList;
ostream& operator<<(ostream& out, const FramesList& obj) {
    out << '[';
    for (size_t i = 0; i < obj.size(); i++) {
        out << obj[i];
        if (i != obj.size() - 1) out << ", ";
    }
    out << ']';
    return out;
}

/**
 * Function: writeBlackFrames
 * --------------------------
 * Writes detected black frames to a file.
 *
 * @param filename the name of the file to write to.
 * @param detected the list of detected black frame IDs.
 *
 * @return true if frames are successfully written, false otherwise.
 */
bool writeBlackFrames(const string& filename, const FramesList& detected);

/**
 * Function: getIOPaths
 * --------------------
 * Retrieves file input and output paths from STDIN.
 *
 * @param videoPaths the list of video file paths to be updated.
 * @param outputPaths the list of output directory paths to be updated.
 */
void getIOPaths(vector<string>& videoPaths, vector<string>& outputPaths);

// Command line arguments
static size_t numVideos = 0;
static size_t numThreads = 1;
static bool force = false;
void getArgs(int argc, char *argv[]) {
    char c;
    while ((c = getopt(argc, argv, "n:j:f")) != -1) {
        switch (c) {
            case 'n': numVideos = stoi(optarg);
                      break;
            case 'j': numThreads = stoi(optarg);
                      break;
            case 'f': force = true;
                      break;
            case '?':
                if (optopt == 'j') cerr << "Option -j requires an argument." << endl;
                else cerr << "Unknown option `-" << optopt << "`." << endl;
                exit(1);
            default: abort();
        }
    }

    if (numVideos == 0) {
        cout << "Number of videos is zero. Exiting" << endl;
        exit(0);
    }
}

int main(int argc, char *argv[]) {
    getArgs(argc, argv);

    vector<string> videoPaths;
    vector<string> outputPaths;
    getIOPaths(videoPaths, outputPaths);

    vector<FramesList> outputs(numVideos);
    vector<thread> threads(numThreads);

    for (size_t videoIdx = 0; videoIdx < numVideos; videoIdx++) {
        string videoPath = videoPaths[videoIdx];
        vector<FramesList> detected(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            threads[i] = thread([videoPath, &detected](size_t i) {
                cv::VideoCapture video(videoPath);

                if (!video.isOpened()) {
                    cerr << "Error opening video file." << endl;
                    return;
                }

                size_t numFrames = video.get(cv::CAP_PROP_FRAME_COUNT);
                size_t frameWidth = video.get(cv::CAP_PROP_FRAME_WIDTH);
                size_t frameHeight = video.get(cv::CAP_PROP_FRAME_HEIGHT);
                size_t numPixels = frameWidth * frameHeight;

                size_t payloadSize = numFrames / numThreads;
                size_t startFrame = payloadSize * i;
                if (i == numThreads - 1) payloadSize += numFrames % numThreads;
                size_t endFrame = startFrame + payloadSize;

                video.set(cv::CAP_PROP_POS_FRAMES, startFrame);

                for (size_t frameNum = startFrame; frameNum < endFrame; frameNum++) {
                    cv::Mat frame;
                    video >> frame;

                    cv::Mat grayFrame;
                    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

                    if (cv::countNonZero(grayFrame) <= 0.01 * numPixels) {
                        detected[i].push_back(frameNum);
                    }
                }
            }, i);

        }
        for (thread& t : threads) t.join();

        for (auto& sub : detected) {
            outputs[videoIdx].insert(end(outputs[videoIdx]), begin(sub), end(sub));
        }

        if (!writeBlackFrames(outputPaths[videoIdx], outputs[videoIdx])) {
            cerr << "Could not open output file '" << outputPaths[videoIdx] << "'." << endl;
        }
    }
}

bool writeBlackFrames(const string& filename, const FramesList& detected) {
    ofstream output(filename);
    if (output.fail()) return false;

    output << detected;
    return true;
}

void getIOPaths(vector<string>& videoPaths, vector<string>& outputPaths) {
    string inPath, outPath;

    for (size_t i = 0; i < numVideos; i++) {
        cin >> inPath >> outPath;
        if (outPath == "") {
            cerr << "Error in file inputs. Exiting" << endl;
            exit(1);
        }

        videoPaths.push_back(inPath);
        outputPaths.push_back(outPath);
    }
}
