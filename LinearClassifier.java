import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.*;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.PrintWriter;
import java.util.*;

public class LinearClassifier {


    public static void main(String[] args){

        try {


            System.out.println("Retrieving dataset...");
            GroupedDataset<String, VFSListDataset<FImage>, FImage> allData =
                    new VFSGroupDataset<FImage>("/home/yoanapaleva/Documents/Computer-Vision/training", ImageUtilities.FIMAGE_READER);

            VFSListDataset<FImage> testing = new VFSListDataset<FImage>("/home/yoanapaleva/Documents/Computer-Vision/testing", ImageUtilities.FIMAGE_READER); //load the test set as VFSListDataset

            HardAssigner<float[], float[], IntFloatPair> assigner =
                    trainQuantiser(GroupedUniformRandomisedSampler.sample(allData, 100));

            FeatureExtractor<DoubleFV, FImage> extractor = new PatchExtractor(assigner);


            HomogeneousKernelMap map = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
            extractor = map.createWrappedExtractor(extractor);


            System.out.println("Creating LibLinearAnnotator...");
            LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
            ann.train(allData);


            PrintWriter writer = new PrintWriter("run2.txt", "UTF-8");

            for (int i = 0; i < testing.size(); i++) {
                final String imageId = testing.getID(i);
                final FImage image = testing.get(i);

                final ClassificationResult<String> clz = ann.classify(image);

                writer.println(imageId + " " + clz.getPredictedClasses().iterator().next());
            }

            writer.close();

        } catch (Exception e){
            e.printStackTrace();
        }
    }

    //Given an image, create a list of size x size patches, taken by a certain step
    private static LocalFeatureList<FloatKeypoint> takePatch(FImage image, int size, int step) {
        LocalFeatureList<FloatKeypoint> patches = new MemoryLocalFeatureList<>();
        for (int y = 0; y <= image.getHeight() - size; y += step) {
            for (int x = 0; x <= image.getWidth() - size; x += step) {

                float[] patch = new float[size * size];
                int counter = 0;
                for (int i = y; i < y + size; i++) {
                    for (int j = x; j < x + size; j++) {
                        patch[counter++] = image.pixels[i][j];
                    }
                }
                FloatKeypoint floatKeypoint = new FloatKeypoint();
                floatKeypoint.vector = patch;
                patches.add(floatKeypoint);
            }
        }

        return patches;
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> dataset) {

        //allKeys is the list of patches extracted from every image in the training set
        List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<>();

        for (FImage image : dataset) {
            allkeys.add(takePatch(image, 8, 4));
        }

        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        DataSource<float[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }



    static class PatchExtractor implements FeatureExtractor<DoubleFV, FImage> {

        HardAssigner<float[], float[], IntFloatPair> assigner;

        public PatchExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {

            //The BagOfVisualWords uses the HardAssigner to assign each patch feature
            //to a visual word and compute the histogram
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

            BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(
                    bovw, 2, 2);

            //The resultant spatial histograms are then appended together and normalised before being returned
            return spatial.aggregate(takePatch(image, 8, 4), image.getBounds()).normaliseFV();
        }
    }

}
