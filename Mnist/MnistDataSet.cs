using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist
{
    public class MnistDataSetGenerator
    {
        private int imageX;
        private int imageY;
        private int numberOfClasses;
        private int numberOfTestVectors;
        private int numberOfTrainVectors;
        private int vectorSize;

        public MnistDataSetGenerator(int numberOfClassees, int vectorSize, int numberOfTrainVectors, int numberOfTestVectors)
        {
            this.numberOfClasses = numberOfClassees;
            this.vectorSize = vectorSize;
            this.numberOfTrainVectors = numberOfTrainVectors;
            this.numberOfTestVectors = numberOfTestVectors;
        }

        public MnistDataSetGenerator(int numberOfClassees, int imageX, int imageY)
        {
            this.numberOfClasses = numberOfClassees;
            this.imageX = imageX;
            this.imageY = imageY;
        }

        public MnistDatasetMlp GetMnistDataSet()
        {
            using (var python = Py.GIL())
            {
                dynamic keras = Py.Import("keras");
                dynamic numpy = Py.Import("numpy");
                PyObject mnistModule = PythonEngine.ModuleFromString("mnistDatasetModule", "from keras.datasets import mnist");
                dynamic mnist = mnistModule.GetAttr("mnist");
                dynamic mnistData = mnist.load_data();
                var result = new MnistDatasetMlp();
                result.TrainX = numpy.divide(mnistData[0][0].reshape(numberOfTrainVectors, vectorSize).astype("float32"), 255f);
                result.TrainY = keras.utils.to_categorical(mnistData[0][1], numberOfClasses);
                result.TestX = numpy.divide(mnistData[1][0].reshape(numberOfTestVectors, vectorSize).astype("float32"), 255f);
                result.TestY = keras.utils.to_categorical(mnistData[1][1], numberOfClasses);
                return result;
            }
        }

        public MnistDatasetCnn GetMnistDataSetForCnnNetwork()
        {
            using(var python = Py.GIL())
            {
                var result = new MnistDatasetCnn();
                dynamic keras = Py.Import("keras");
                dynamic numpy = Py.Import("numpy");
                PyObject mnistModule = PythonEngine.ModuleFromString("mnistDatasetModule", "from keras.datasets import mnist");
                PyObject backendModule = PythonEngine.ModuleFromString("backendModule", "from keras import backend");
                dynamic k = backendModule.GetAttr("backend");
                dynamic mnist = mnistModule.GetAttr("mnist");
                dynamic mnistData = mnist.load_data();

                result.TrainX = mnistData[0][0];
                result.TrainY = mnistData[0][1];
                result.TestX = mnistData[1][0];
                result.TestY = mnistData[1][1];

                if (k.image_data_format() == new PyString("channels_first"))
                {
                    result.TrainX = result.TrainX.reshape(result.TrainX.shape[0], 1, imageX, imageY);
                    result.TestX = result.TestX.reshape(result.TestX.shape[0], 1, imageX, imageX);
                    result.InputShape = new PyTuple(new PyObject[] { new PyInt(1), new PyInt(imageX), new PyInt(imageY) });
                }
                else
                {
                    result.TrainX = result.TrainX.reshape(result.TrainX.shape[0], imageX, imageY, 1);
                    result.TestX = result.TestX.reshape(result.TestX.shape[0], imageX, imageY, 1);
                    result.InputShape = new PyTuple(new PyObject[] { new PyInt(imageX), new PyInt(imageY), new PyInt(1) });
                }

                result.TrainX = numpy.divide(result.TrainX.astype("float32"), 255f);
                result.TestX = numpy.divide(result.TestX.astype("float32"), 255f);
                result.TrainY = keras.utils.to_categorical(result.TrainY, numberOfClasses);
                result.TestY = keras.utils.to_categorical(result.TestY, numberOfClasses);

                return result;
            }
        }

        
    }

    public class MnistDatasetMlp
    {
        public dynamic TrainX { get; set; }

        public dynamic TrainY { get; set; }

        public dynamic TestX { get; set; }

        public dynamic TestY { get; set; }
    }

    public class MnistDatasetCnn : MnistDatasetMlp
    {
        public PyTuple InputShape { get; set; }
    }
}
