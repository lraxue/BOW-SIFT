

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iomanip>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Step 1 - Obtain the set of bags of featues

    char* filename = new char[100];
    Mat input;

    vector<KeyPoint> keypoints;
    Mat descrptor;
    Mat featuresUnclustered;

    SiftFeatureDetector detector;
    SiftDescriptorExtractor extractor;
//
//    for (int f = 0; f < 100; f++)
//    {
//        stringstream ss;
//        ss << setfill('0') << setw(6) << f;
//
//        input = imread("./images/" + ss.str() + ".png", CV_LOAD_IMAGE_GRAYSCALE);
//
//        if (input.empty())
//        {
//            cout << "Image " << ss.str() << " is empty." << endl;
//            return -1;
//        }
//
//        detector.detect(input, keypoints);
//
//        extractor.compute(input, keypoints, descrptor);
//
//        featuresUnclustered.push_back(descrptor);
//
//        printf("%f percent done\n", f/100.0);
//    }
//
//    cout << "End of loading training images." << endl;
//    // Number of bags
//    int dictionarySize = 200;
//    // Define term criteria
//    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
//    // Retries number
//    int retries = 1;
//    // Necessary flags
//    int flags = KMEANS_PP_CENTERS;
//    // Create the BoW (BoF) trainer
//    BOWKMeansTrainer bowkMeansTrainer(dictionarySize, tc, retries, flags);
//    // Cluster the feature vectors
//    Mat dictionary = bowkMeansTrainer.cluster(featuresUnclustered);
//    // Store the vocabulary
//    FileStorage fs("./results/dictionary.yml", FileStorage::WRITE);
//    fs << "vocabulary" << dictionary;
//    fs.release();
//
//    cout << "End of generating dictionary." << endl;

    // Step 2 - Obtain the BoF descriptor for given image/video frame.
    // Prepare BoW descriptor extractor from the dictionary
    Mat vocabulary;
    FileStorage fv("./results/dictionary.yml", FileStorage::READ);
    fv["vocabulary"] >> vocabulary;
    fv.release();

    // Create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    // Create SIFT feature point extractor
    Ptr<FeatureDetector> featureDetector(new SiftFeatureDetector);
    // Create SIFT feature descriptor
    Ptr<DescriptorExtractor> descriptorExtractor(new SiftDescriptorExtractor);
    // Create BoW descriptor extractor
    BOWImgDescriptorExtractor bowDE(descriptorExtractor, matcher);
    // Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(vocabulary);

    // To store image tag name
    char* imageTag = new char[10];

    // Open the file to write the result descriptor
    FileStorage fs1("./results/descriptor.yml", FileStorage::WRITE);

    sprintf(filename, "./images/000100.png");
    // Read the image
    Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    // featureDetector->detect(img, keypoints);
    detector.detect(img, keypoints);
    Mat bowDescriptor;
    vector<vector<int> > indices;
    bowDE.compute(img, keypoints, bowDescriptor, &indices);

    // Prepare yml
    sprintf(imageTag, "img1");
    fs1 << imageTag << bowDescriptor;
    fs1 << "Indices" << indices;

    fs1.release();

    printf("\ndone\n");


    return 0;
}