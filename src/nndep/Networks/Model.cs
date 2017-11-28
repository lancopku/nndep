using System.Collections.Generic;
using nndep.Networks.Graph;
using System;
using System.Runtime.Serialization;

namespace nndep.Networks
{
    [Serializable]
    public abstract class Model
	{
        [NonSerialized] protected List<Tensor> AllParams;
        [NonSerialized] protected List<Tensor> FixedParams;
        [NonSerialized] protected Graph.Graph G;
        [NonSerialized] protected HashSet<Tensor> VariedParams;


		protected Model()
		{
			AllParams = new List<Tensor>();
			FixedParams = new List<Tensor>();
			VariedParams = new HashSet<Tensor>();
			G = new Graph.Graph();
		}

        [OnDeserialized()]
        internal void OnDeserializedMethod(StreamingContext context)
        {
            AllParams = new List<Tensor>();
            FixedParams = new List<Tensor>();
            VariedParams = new HashSet<Tensor>();
            G = new Graph.Graph();
        }
    }
}