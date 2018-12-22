import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;

import java.awt.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class kNearestNeighbours {


    public static void main(String[] args) throws IOException {

        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("/home/yoanapaleva/Documents/Computer-Vision/training", ImageUtilities.FIMAGE_READER);

        VFSListDataset<FImage> test = new VFSListDataset<FImage>("/home/yoanapaleva/Documents/Computer-Vision/testing", ImageUtilities.FIMAGE_READER);

        int size = 16;
        int k = 3;

        HashMap<String, float[]> testVectors = preprocessAlltoMap(test, size, test.size());

        HashMap<String, float[][]> trainingVectors = new HashMap<>();

        //process each class of the training dataset
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : training.entrySet()) {
            trainingVectors.put(entry.getKey(), preprocessAlltoMatrix(entry.getValue(), size, entry.getValue().size()));
        }

        PrintWriter writer = new PrintWriter("run1.txt", "UTF-8");


        //iterate over each image in the test dataset
        for (Map.Entry<String, float[]> testImage : testVectors.entrySet()) {


            ArrayList<Neighbour> neighbours = new ArrayList<>();
            //iterate over each class of the training dataset
            for (Map.Entry<String, float[][]> entry : trainingVectors.entrySet()) {

                //iterate over each image of that class
                for (float[] vector : entry.getValue()) {
                    float distance = euclideanDistance(testImage.getValue(), vector);
                    update(neighbours, new Neighbour(entry.getKey(), distance), k);
                }

            }
            writer.println(testImage.getKey() + " " + classify(neighbours));
        }


        writer.close();


    }

    //crop the image to a square around the center
    public static void cropImage(FImage image) {

        int width = image.getWidth();
        int height = image.getHeight();

        int squareSize = 0;

        //set the size of the square to the smaller side
        if (width > height) {
            squareSize = height;
        } else {
            squareSize = width;
        }

        FImage result = new FImage(squareSize, squareSize);
        //calculate the upper-left starting point for cropping
        Point startPoint = new Point((width - squareSize) / 2, (height - squareSize) / 2);

        //save the cropped image into the result
        for (int y = 0; y < squareSize; y++) {
            for (int x = 0; x < squareSize; x++) {
                result.pixels[y][x] = image.pixels[(int) startPoint.getY() + y][(int) startPoint.getX() + x];

            }
        }

        image.internalAssign(result);
    }

    public static void resizeImage(FImage image, int size) {
        image.internalAssign(ResizeProcessor.resample(image, size, size));
    }

    //return vector from an image pixel values
    private static float[] vectorize(FImage image, int size) {
        float[] vector = new float[size * size];
        int counter = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                vector[counter++] = image.pixels[i][j];
            }
        }

        return vector;
    }

    //preprocess a dataset of images into float matrix
    //preprocessing involves cropping, resizing and vectorising an image
    private static float[][] preprocessAlltoMatrix(VFSListDataset<FImage> list, int size, int datasetSize) {
        float[][] vectors = new float[datasetSize][size * size];
        for (int i = 0; i < datasetSize; i++) {
            FImage image = list.getInstance(i);
            cropImage(image);
            resizeImage(image, size);
            FloatFV floatFV = new FloatFV(vectorize(image, size));
            floatFV.normaliseFV();
            vectors[i] = floatFV.getVector();
            zeroMean(vectors[i]);
        }
        return vectors;
    }

    //preprocess a dataset of images into map
    //map key is the image name and map value is the vector of the image
    //preprocessing involves cropping, resizing and vectorising an image
    private static HashMap<String, float[]> preprocessAlltoMap(VFSListDataset<FImage> list, int size, int datasetSize) {
        HashMap<String, float[]> vectors = new HashMap<>();
        for (int i = 0; i < datasetSize; i++) {
            FImage image = list.getInstance(i);
            cropImage(image);
            resizeImage(image, size);
            FloatFV floatFV = new FloatFV(vectorize(image, size));
            floatFV.normaliseFV();
            vectors.put(list.getID(i), floatFV.getVector());
            zeroMean(vectors.get(list.getID(i)));

        }
        return vectors;
    }


    //return the euclidean distance between two vectors
    private static float euclideanDistance(float[] v1, float[] v2) {
        float distance = 0.0f;

        for (int i = 0; i < v1.length; i++) {
            distance += Math.pow((v1[i] - v2[i]), 2);
        }

        return (float) Math.sqrt(distance);
    }


    //sort arraylist of neighbours in ascending way
    private static void sortByDistance(ArrayList<Neighbour> neighbours) {

        Collections.sort(neighbours, new Comparator<Neighbour>() {
            @Override
            public int compare(Neighbour neighbour1, Neighbour neighbour2) {
                if (neighbour1.distance < neighbour2.distance) {
                    return -1;
                } else if (neighbour1.distance > neighbour2.distance) {
                    return 1;
                } else {
                    return 0;
                }
            }
        });
    }

    //update the k nearest neighbours if the new neighbour is in the k nearest
    private static void update(ArrayList<Neighbour> neighbours, Neighbour newNeighbour, int k) {

        //add the neighbour if the list contains less than k neighbours
        //sort the list after adding a new one
        if (neighbours.size() < k) {
            neighbours.add(newNeighbour);
            sortByDistance(neighbours);
        } else {
            //compare the new neighbour's distance to the existing distances in the list
            boolean check = false;
            if (neighbours.get(neighbours.size() - 1).distance > newNeighbour.distance) {
                //when a smaller distance is found insert the new neighbour after it
                for (int i = neighbours.size() - 2; i >= 0; i--) {
                    if (neighbours.get(i).distance < newNeighbour.distance) {
                        neighbours.add(i + 1, newNeighbour);
                        check = true;
                        break;
                    }
                }
                //add the new neighbour at the head of the list if no neighbour in the list has a smaller distance
                if (!check) {
                    neighbours.add(0, newNeighbour);
                }
                //remove the k+1 nearest from the list
                neighbours.remove(neighbours.size() - 1);
            }
        }
    }

    //classify an image given its k nearest neighbours
    private static String classify(ArrayList<Neighbour> neighbours) {
        HashMap<String, Integer> occurrences = new HashMap<>();

        //count the occurrences of each class in the k nearest neighbours
        for (int i = 0; i < neighbours.size(); i++) {
            if (!occurrences.containsKey(neighbours.get(i).label)) {
                occurrences.put(neighbours.get(i).label, 1);
            } else {
                occurrences.put(neighbours.get(i).label, occurrences.get(neighbours.get(i).label) + 1);
            }
        }

        int maxOccurrences = Collections.max(occurrences.values());

        //make a list of the most frequent occurrences
        ArrayList<String> mostFrequent = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : occurrences.entrySet()) {
            if (entry.getValue().equals(maxOccurrences)) {
                mostFrequent.add(entry.getKey());
            }
        }

        //if more than one class have equal number of max occurrences, choose the class with least total distance
        if (mostFrequent.size() > 1) {
            float minDistance = totalDistance(neighbours, mostFrequent.get(0));
            String minLabel = mostFrequent.get(0);
            for (int i = 1; i < mostFrequent.size(); i++) {
                float currentDistance = totalDistance(neighbours, mostFrequent.get(i));
                if (currentDistance < minDistance) {
                    minDistance = currentDistance;
                    minLabel = mostFrequent.get(i);
                }
            }
            return minLabel;
            //if only one class has the most frequent distance, return it
        } else {
            return mostFrequent.get(0);
        }
    }

    //calculate the total distance of a label in the k nearest neighbours
    private static float totalDistance(ArrayList<Neighbour> neighbours, String label) {
        float sum = 0.0f;
        for (int i = 0; i < neighbours.size(); i++) {
            if (neighbours.get(i).label.equals(label)) {
                sum += neighbours.get(i).distance;
            }
        }
        return sum;
    }

    //process a vector to have a zero mean
    private static void zeroMean(float[] vector) {
        float sum = 0.0f;
        for (int i = 0; i < vector.length; i++) {
            sum += vector[i];
        }

        float mean = sum / vector.length;
        for (int i = 0; i < vector.length; i++) {
            vector[i] = vector[i] - mean;
        }
    }

    //class to store neighbour data
    private static class Neighbour {

        protected String label;
        protected float distance;

        public Neighbour(String label, float distance) {
            this.label = label;
            this.distance = distance;
        }

    }

}

