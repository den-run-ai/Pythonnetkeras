using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist.Constants
{
    public static class KerasConstants
    {
        public static class LossFunction
        {
            public const string CategoricalCrossentropy = "categorical_crossentropy";
        }

        public static class ActivationFunction
        {
            public const string Adadelta = "Adadelta";
            public const string Relu = "relu";
            public const string Softmax = "softmax";
            public const string RmsProp = "RMSprop";
        }

        public static class Metrics
        {
            public const string Accuracy = "accuracy";
        }

        public static class Compile
        {
            public const string Loss = "loss";

            public const string Optimizer = "optimizer";

            public const string Metrics = "metrics";
        }

        public static class Evaluate
        {
            public const string Verbose = "verbose";
        }

        public static class Fit
        {
            public const string BatchSize = "batch_size";

            public const string Epochs = "epochs";

            public const string Verbose = "verbose";

            public const string ValidationData = "validation_data";
        }

        public static class DataTypes
        {
            public const string Float32 = "float32";
        }

        public static class Layer
        {
            public const string Activation = "activation";
            public const string InputShape = "input_shape";
            public const string PoolSize = "pool_size";
        }
    }
}
