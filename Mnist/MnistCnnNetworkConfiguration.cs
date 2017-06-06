using Mnist.Constants;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Python.Runtime.Py;

namespace Mnist
{
    public class MnistCnnNetworkConfiguration : IDisposable
    {
        private GILState python;
        private dynamic model;
        private dynamic convolutional2DLayer;
        private dynamic denseLayer;
        private dynamic dropoutLayer;
        private dynamic maxPooling2DLayer;
        private dynamic flattenLayer;

        public MnistCnnNetworkConfiguration()
        {
            this.python = Py.GIL();
            ConfigureModules();
        }

        public MnistCnnNetworkConfiguration AddMaxPooling2DLayer(PyTuple poolSize)
        {
            model.add(maxPooling2DLayer(Py.kw(KerasConstants.Layer.PoolSize, poolSize)));
            return this;
        }

        public MnistCnnNetworkConfiguration AddConvolutional2DLayer(int filterSize, PyTuple kernelSize, string activation, PyTuple inputShape = null)
        {
            dynamic convolutionalLayer = null;
            if(inputShape != null)
            {
                convolutionalLayer = convolutional2DLayer(filterSize, kernelSize, Py.kw(KerasConstants.Layer.Activation, activation, KerasConstants.Layer.InputShape, inputShape));
            }
            else
            {
                convolutionalLayer = convolutional2DLayer(filterSize, kernelSize, Py.kw(KerasConstants.Layer.Activation, activation));
            }

            model.add(convolutionalLayer);
            return this;
        }


        public MnistCnnNetworkConfiguration AddDropoutLayer(double ratio)
        {
            model.add(dropoutLayer(ratio));
            return this;
        }

        public MnistCnnNetworkConfiguration AddFlattenLayer()
        {
            model.add(flattenLayer());
            return this;
        }

        public MnistCnnNetworkConfiguration AddDenseLayer(int numberOfNeurons, string activationFunction, PyTuple inputShape = null)
        {
            dynamic denseNetworkLayer = null;
            if (inputShape != null)
            {
                denseNetworkLayer = denseLayer(numberOfNeurons, Py.kw(KerasConstants.Layer.Activation, activationFunction, KerasConstants.Layer.InputShape, inputShape));
            }
            else
            {
                denseNetworkLayer = denseLayer(numberOfNeurons, Py.kw(KerasConstants.Layer.Activation, activationFunction));
            }

            model.add(denseNetworkLayer);
            return this;
        }

        public string Build()
        {
            return model.to_json();
        }

        public void Dispose()
        {
            this.python.Dispose();
        }

        private void ConfigureModules()
        {
            PyObject SequentialModule = PythonEngine.ModuleFromString("sequential", "from keras.models import Sequential");
            PyObject layersModule = PythonEngine.ModuleFromString("standardLayers", "from keras.layers import Dense, Dropout, Flatten");
            PyObject layersCnnModule = PythonEngine.ModuleFromString("conEvoLayers", "from keras.layers import Conv2D, MaxPooling2D");

            dynamic sequential = SequentialModule.GetAttr("Sequential");
            this.model = sequential();
            this.denseLayer = layersModule.GetAttr("Dense");
            this.dropoutLayer = layersModule.GetAttr("Dropout");
            this.flattenLayer = layersModule.GetAttr("Flatten");
            this.convolutional2DLayer = layersCnnModule.GetAttr("Conv2D");
            this.maxPooling2DLayer = layersCnnModule.GetAttr("MaxPooling2D");
        }
    }
}
