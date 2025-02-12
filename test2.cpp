#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include "lstm.h"

using namespace std;

#define INPUT   6  // Modify this based on your input features
#define OUTPUT  1  // Modify this based on your output features
#define HIDE    12  // You can adjust hidden layer size
#define FOR(i,n) for(int i=0;i<n;i++)

// Function to load data from CSV file
bool loadDataFromCSV(const string& filename, vector<double*>& features, vector<double*>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file: " << filename << endl;
        return false;
    }

    string line;
    // Skip header if exists
    getline(file, line);

    while (getline(file, line)) {
        double* input = (double*)malloc(sizeof(double) * INPUT);
        double* label = (double*)malloc(sizeof(double) * OUTPUT);
        
        // Parse CSV line
        size_t pos = 0;
        string token;
        int column = 0;
        
        while ((pos = line.find(",")) != string::npos) {
            token = line.substr(0, pos);
            
            // Assuming first INPUT columns are features and last OUTPUT columns are labels
            if (column < INPUT) {
                input[column] = stod(token);
            } else {
                label[column - INPUT] = stod(token);
            }
            
            line.erase(0, pos + 1);
            column++;
        }
        // Handle last column
        if (column < INPUT + OUTPUT) {
            if (column < INPUT) {
                input[column] = stod(line);
            } else {
                label[column - INPUT] = stod(line);
            }
        }

        features.push_back(input);
        labels.push_back(label);
    }

    file.close();
    return true;
}

// Function to normalize data (Min-Max scaling)
void normalizeData(vector<double*>& data, int feature_size, double* min_vals, double* max_vals) {
    // First pass: find min and max values
    FOR(j, feature_size) {
        min_vals[j] = data[0][j];
        max_vals[j] = data[0][j];
        
        FOR(i, data.size()) {
            if (data[i][j] < min_vals[j]) min_vals[j] = data[i][j];
            if (data[i][j] > max_vals[j]) max_vals[j] = data[i][j];
        }
    }

    // Second pass: normalize
    FOR(i, data.size()) {
        FOR(j, feature_size) {
            if (max_vals[j] != min_vals[j]) {
                data[i][j] = (data[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j]);
            } else {
                data[i][j] = 0.5; // Handle constant features
            }
        }
    }
}

int main() {
    vector<double*> trainSet;
    vector<double*> labelSet;
    
    // Load data from CSV file
    if (!loadDataFromCSV("traffic2.csv", trainSet, labelSet)) {
        return 1;
    }

    // Normalize features
    double* feature_min = new double[INPUT];
    double* feature_max = new double[INPUT];
    double* label_min = new double[OUTPUT];
    double* label_max = new double[OUTPUT];

    normalizeData(trainSet, INPUT, feature_min, feature_max);
    normalizeData(labelSet, OUTPUT, label_min, label_max);

    // Initialize LSTM
    Lstm* lstm = new Lstm(INPUT, HIDE, OUTPUT);

    // Train the model
    cout << "Starting training..." << endl;
    lstm->train(trainSet, labelSet, 1000, 0, 0.001); // Adjust epochs and learning rate as needed

    // Test the model with some sample data 
    cout << "\nTesting model with sample data:" << endl;
    FOR(i, 5) { // Test with 5 samples
        if (i < trainSet.size()) {
            double* prediction = lstm->predict(trainSet[i]);
            
            // Denormalize prediction            
            FOR(j, OUTPUT) {
                prediction[j] = prediction[j] * (label_max[j] - label_min[j]) + label_min[j];
                labelSet[i][j] = labelSet[i][j] * (label_max[j] - label_min[j]) + label_min[j];
            }
            cout << "Sample " << i + 1 << ":" << endl;
            cout << "Predicted: " << prediction[0] << ", Actual: " << labelSet[i][0] << endl;
            cout << "Difference: " << abs(prediction[0] - labelSet[i][0]) << endl << endl;
        }
    }

    lstm->ShowWeights();
    // Cleanup
    delete lstm;
    delete[] feature_min;
    delete[] feature_max;
    delete[] label_min;
    delete[] label_max;
    
    FOR(i, trainSet.size()) {
        free(trainSet[i]);
        free(labelSet[i]);
    }
    trainSet.clear();
    labelSet.clear();

    return 0;
}
