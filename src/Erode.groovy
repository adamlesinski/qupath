import org.bytedeco.opencv.opencv_core.*
import qupath.lib.awt.common.BufferedImageTools
import qupath.lib.common.ColorTools
import qupath.lib.images.ImageData
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathROIObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs
import qupath.lib.roi.interfaces.ROI
import qupath.opencv.tools.OpenCVTools

import java.awt.image.BufferedImage

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1
import static org.bytedeco.opencv.global.opencv_core.convertScaleAbs
import static org.bytedeco.opencv.global.opencv_imgproc.*

// --------------------------
// BEGIN: Parameters to tune.
// --------------------------

// How much to lower the resolution of the image when processing.
// Lowering this increases the resolution of the resulting detection
// polygons, but will also add more noise to the perimeter.
// Increasing this will make the calculations run faster, but you lose
// detail and the resulting polygons may overlap muscle fibers.
final double DOWN_SAMPLE = 2.0

// Which channel of the image to process, starting from 0.
final int CHANNEL = 0

// The color to use for the detections in the QuPath editor.
final Integer ANNOTATION_COLOR = ColorTools.packRGB(0, 200, 50);

// ------------------------
// END: Parameters to tune.
// ------------------------

// Converts a 16 bit grayscale image to 8 bit. Colors are normalized based on
// the brightest color in the masked portion of the image.
static Mat narrowFrom16To8(Mat sourceImage, Mat mask) {
    // Compute the brightest color in the source image and use that as the max value when
    // normalizing the values from 0 - 255.
    double largestValue = OpenCVTools.extractMaskedDoubles(sourceImage, mask, 0).max()
    def output = new Mat(sourceImage.size(), CV_8UC1);
    convertScaleAbs(sourceImage, output, (1.0 / largestValue) * 255.0, 0.0)
    return output
}

// Find the ROI from the largest contour of the `sourceImage`. The new ROI is
// defined from `roi`, with the same `downSample` that occurred on `sourceImage`.
static ROI findRoiFromContour(Mat sourceImage, ROI roi, double downSample) {
    def contours = new MatVector()
    findContours(sourceImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE)
    if (contours.size() == 0) {
        throw new Exception("No contour found for detected region :(")
    }
    long largestIndex = 0;
    for (long i = 1; i < contours.size(); i++) {
        if (contours.get(i).rows() > contours.get(largestIndex).rows()) {
            largestIndex = i;
        }
    }

    def coords = OpenCVTools.extractDoubles(contours.get(largestIndex))

    // The x, y coordinates are packed side by side, so x1, y1, x2, y2, ...
    def xs = new double[coords.length / 2];
    def ys = new double[coords.length / 2];
    for (int i = 0; i < coords.length / 2; i += 1) {
        xs[i] = coords[i * 2] * downSample + roi.getBoundsX();
        ys[i] = coords[i * 2 + 1] * downSample + roi.getBoundsY();
    }
    return ROIs.createPolygonROI(xs, ys, roi.getImagePlane())
}

// Takes an original detection ROI and returns an eroded copy of the detection that excludes
// any muscle fibers.
PathDetectionObject erodeDetection(
        ImageData<BufferedImage> imageData,
        int channel,
        double downSample,
        PathROIObject pathObject
) {
    def roi = pathObject.getROI()

    // Read image region defined by the ROI
    def request = RegionRequest.createInstance(
            imageData.getServerPath(),
            downSample,
            roi)
    def sourceImg = OpenCVTools.imageToMat(imageData.getServer().readRegion(request))

    // The ROI is actually a polygon, so we need to create an ROI mask
    def mask = OpenCVTools.imageToMat(
            BufferedImageTools.createROIMask(
                    sourceImg.size().width(),
                    sourceImg.size().height(),
                    roi,
                    request))
    def maskInv = mask.clone()
    OpenCVTools.invertBinary(mask, maskInv)

    // Extract the channel we're interested in.
    def sourceImageChannels = OpenCVTools.splitChannels(sourceImg)
    if (channel >= sourceImageChannels.size()) {
        throw new Exception("`channel` ${channel} does not exist. There are only ${sourceImageChannels.size()} channels in the source image.")
    }
    sourceImage = sourceImageChannels.get(channel)

    // Assuming the source image has a bit depth of 16, we need to narrow this to 8 for
    // most OpenCV algorithms to work.
    def outputImage = narrowFrom16To8(sourceImage, mask)

    // Blur the image, so that the muscle fibers expand their borders a bit,
    // making for a more conservative boundary.
    GaussianBlur(outputImage, outputImage, new Size(15, 15), 0)

    // Threshold the image to end up with a binary image.
    // Our goal is for the muscle fibers to turn white, and everything else
    // to be black.
    threshold(outputImage, outputImage, 100, 255, THRESH_BINARY + THRESH_OTSU)

    // Color all the area outside the mask white.
    OpenCVTools.fill(outputImage, maskInv, 255.0)

    // Flood fill from the centroid of our region of interest.
    // The goal is to start in the black area and fill until we reach the white border
    // regions. This new 'gray' area will be our detection region.
    def localCentroid = new Point(
            (int) ((roi.getCentroidX() - roi.getBoundsX()).round().toLong() / downSample),
            (int) ((roi.getCentroidY() - roi.getBoundsY()).round().toLong() / downSample))
    floodFill(outputImage, localCentroid, new Scalar(150))

    // Cut out the flood-filled gray area into its own black and white mask,
    // with the region we are interested in colored white.
    def regionMask = OpenCVTools.createBinaryMask(outputImage, { it -> it == 150.0 })

    // QuPath wants a polygon, made of points, to represent the detection area.
    // We detect the contours of our mask and select the largest polygon found,
    // since smaller ones could just be contours of noise in the mask.
    def erodedRoi = findRoiFromContour(regionMask, roi, downSample)

    // Create a detection object and return that.
    return PathObjects.createDetectionObject(erodedRoi) as PathDetectionObject
}

// Gathers the selected detections in the QuPath editor, descending into the
// object hierarchy and gathering any child detection objects.
// Returns a flat list of detections to process.
def gatherSelectedDetections() {
    // Use a set to ensure uniqueness, which is an issue when the user
    // selected a root annotation object and one of its children.
    def detections = new HashSet()
    getSelectedObjects().each {
        if (it.isDetection()) {
            // This is a detection, add it to the list.
            detections.add(it)
        } else {
            // This is not a detection, so see if any children of this object are
            // detections.
            def childDetections = it.getDescendantObjects(null).findAll { it.isDetection() }
            detections.addAll(childDetections)
        }
    }
    return detections.toList()
}

// Finds the root annotation, from which the high level ROI can be copied for
// the new detections.
def findRootAnnotation() {
    def annotations = new HashSet()
    getSelectedObjects().each {
        // Walk up the hierarchy (towards the root, following parent nodes)
        // to find the first parent annotation (not detection).
        def current = it
        while (current != null && !current.isAnnotation()) {
            current = current.getParent()
        }
        if (current != null) {
            // This is the parent annotation, add it to the list.
            annotations.add(current)
        }
    }
    if (annotations.size() > 1) {
        throw new Exception("More than one root annotation selected, only select one")
    } else if (annotations.size() == 0) {
        throw new Exception("No root annotation selected")
    } else {
        return annotations.iterator().next()
    }
}

// Create a root annotation to hold all the eroded detections we generate.
// Base it off the original root annotation's ROI.
def rootAnnotation = PathObjects.createAnnotationObject(findRootAnnotation().getROI())
rootAnnotation.setName("Eroded Detections")
rootAnnotation.setColor(ANNOTATION_COLOR)

// Process each detection
def detectionsToProcess = gatherSelectedDetections()
final int totalDetections = detectionsToProcess.size()
detectionsToProcess.eachWithIndex { original, idx ->
    print "Eroding detection ${idx + 1}/${totalDetections}..."
    def detection = erodeDetection(getCurrentImageData(), CHANNEL, DOWN_SAMPLE, original)
    detection.setColor(ANNOTATION_COLOR)
    rootAnnotation.addChildObject(detection)
}

// Now that we're done adding to it, lock the root annotation so the user doesn't
// accidentally move them around in the editor.
rootAnnotation.setLocked(true)

// Add the root annotation (and consequentially all detections added to it) to
// the path object hierarchy attached to the image.
addObject(rootAnnotation)

print "Done"