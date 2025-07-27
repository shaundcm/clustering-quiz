# QuizCluster
## Overview
QuizCluster is a Python-based system that clusters people based on their quiz responses using K-means clustering. It implements Principal Component Analysis (PCA) for dimensionality reduction and anomaly detection based on cluster distances. The system uses NumPy and Pandas for data processing and Plotly for interactive visualizations, enabling intuitive analysis of user response patterns.

## Features
- Data Loading: Reads quiz response data from a CSV file, with names in the first column and numerical responses in subsequent columns.
- Dimensionality Reduction: Applies PCA to reduce data to 1-3 dimensions for clustering and visualization.
- K-Means Clustering: Groups responses into user-specified clusters using the K-means algorithm.
- Anomaly Detection: Identifies outliers based on distance from cluster centroids with a configurable threshold.
- Visualization: Generates interactive 1D, 2D, or 3D scatter plots using Plotly, with hover-over names for easy identification.
- Output: Displays cluster assignments and anomalies in a clear, text-based format.

## Requirements

- Python 3.x
- Libraries: numpy, pandas, plotly.express
- Install dependencies using:
```bash
pip install numpy pandas plotly
```

## Usage

### Prepare Input Data:
Create a CSV file with quiz responses.
### Format: 
First column contains names (strings), remaining columns contain numerical responses.
#### Example:
```
Alice,5,3,4
Bob,2,4,5
Charlie,1,2,3
```
### Run the Program:

Execute the script in a Python environment:
```bash
python quiz_cluster.py
```

### Follow prompts to:
- Enter the CSV file path.
- Specify the number of dimensions (1-3) for PCA reduction.
- Specify the number of clusters for K-means.

## Output:

### Console displays:
- Cluster assignments for each person.
- Detected anomalies with their assigned clusters.

Interactive Plotly visualization (1D, 2D, or 3D) opens in a browser.

#### Example:
```
Enter the path to the CSV file: quiz_data.csv
Enter how many dimensions you want to reduce it to (1-3): 2
Enter the number of clusters: 3

Output: Text listing clusters and anomalies, plus an interactive 2D scatter plot.
```
## Notes

- Ensure the CSV file is properly formatted to avoid errors.
- The anomaly detection threshold is set to 7 (hardcoded) but can be adjusted in the detect_anomalies function.
- Visualizations require a browser to display Plotly graphs.
- The program assumes numerical data in the CSV (except for the name column).

## License
This project is licensed under the MIT License.

## Authors

- [Deishaun Colins](https://github.com/shaundcm)
- [M Deshna Shree](https://github.com/dsdeshna)
