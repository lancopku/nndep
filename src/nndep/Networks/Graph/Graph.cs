using System;
using System.Collections.Generic;
using System.Linq;
using nndep.Util;

namespace nndep.Networks.Graph
{
	public class Graph
	{
		private List<Action> _backwardOps;
		public bool Need;
	    private RandomNumberGenerator _rng;

		public Graph(RandomNumberGenerator rng=null)
		{
			_backwardOps = new List<Action>();
			Need = false;
		    _rng = rng;
		}

	    public void SetRng(int seed)
	    {
	        _rng = new RandomNumberGenerator(seed);
	    }

		public Tensor Add(Tensor l, Tensor r)
		{
			var res = new Tensor(l.Row, l.Col, Need);
			var len = res.W.Storage.Length;
			// unsafe 
			// {
			// 	fixed(float* y = res.W.Storage, a = l.W.Storage, b = r.W.Storage){
			// 		float * py = y;
			// 		float * pa = a;
			// 		float * pb = b;
			// 		for(var i = 0; i < len; i++)
			// 		{
			// 			py[i] = pa[i] + pb[i];
			// 		}
			// 	}
			// }
			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = l.W.Storage[i] + r.W.Storage[i];
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					l.Grad.Storage[i] += res.Grad.Storage[i];
					r.Grad.Storage[i] += res.Grad.Storage[i];
				}
			});
			return res;
		}


		public Tensor AddBias(Tensor param, Tensor bias)
		{
			var nBias = bias.Col;
			var res = new Tensor(param.Row, param.Col, Need);
            bias.RefCount++;

			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = param.W.Storage[i] + bias.W.Storage[i % nBias];
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					param.Grad.Storage[i] += res.Grad.Storage[i];
					bias.Grad.Storage[i % nBias] += res.Grad.Storage[i];
				}
			});
			return res;
		}

		public void Backward()
		{
			for (var i = _backwardOps.Count - 1; i >= 0; --i)
			{
				_backwardOps[i]();
			}
		}

		public void Clear()
		{
			_backwardOps.Clear();
		}

		public Tensor Concat(params Tensor[] list)
		{
			// for row vector only
			//var nTensor = list.Length;
			var nCol = list.Sum(t => t.Col);
			var res = new Tensor(1, nCol, Need);
			for (int t = 0, i = 0; t < list.Length; t++)
			{
				for (var j = 0; j < list[t].W.Storage.Length; j++, i++)
				{
					res.W.Storage[i] = list[t].W.Storage[j];
				}
			}

			if (!Need)
			{
				return res;
			}

			_backwardOps.Add(delegate
			{
				for (int t = 0, i = 0; t < list.Length; t++)
				{
					for (var j = 0; j < list[t].W.Storage.Length; j++, i++)
					{
						list[t].Grad.Storage[j] += res.Grad.Storage[i];
					}
				}
			});
			return res;
		}

		public Tensor Concat(Tensor lhs, Tensor rhs)
		{
			var nCol = lhs.Col + rhs.Col;
			var res = new Tensor(1, lhs.Col + rhs.Col, Need);

			for (var i = 0; i < lhs.Col; i++)
			{
				res.W.Storage[i] = lhs.W.Storage[i];
			}

			for (var i = lhs.Col; i < nCol; i++)
			{
				res.W.Storage[i] = rhs.W.Storage[i - lhs.Col];
			}

			if (!Need)
			{
				return res;
			}

			_backwardOps.Add(delegate
			{
				for (var i = 0; i < lhs.Col; i++)
				{
					lhs.Grad.Storage[i] += res.Grad.Storage[i];
				}

				for (var i = lhs.Col; i < nCol; i++)
				{
					rhs.Grad.Storage[i - lhs.Col] += res.Grad.Storage[i];
				}
			});
			return res;
		}

		public Tensor Cubic(Tensor t)
		{
			var res = new Tensor(t.Row, t.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = t.W.Storage[i] * t.W.Storage[i] * t.W.Storage[i];
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					t.Grad.Storage[i] += 2 * t.W.Storage[i] * t.W.Storage[i] * res.Grad.Storage[i];
				}
			});
			return res;
		}

		public Tensor DropOut(Tensor t, float prop)
		{
			// only apply dropout when training
			if (!Need)
			{
				return t;
			}
			var indices = Enumerable.Range(0, t.W.Storage.Length).Where(n => _rng.GetFloat() > prop).ToArray();
			var res = new Tensor(t.Row, t.Col, Need);
		    var factor = 1.0f / (1.0f - prop);
			foreach (var index in indices)
			{
				res.W.Storage[index] = t.W.Storage[index] * factor;
			}
			_backwardOps.Add(delegate
			{
				foreach (var index in indices)
				{
					t.Grad.Storage[index] += res.Grad.Storage[index] * factor;
				}
			});
			return res;
		}

		public Tensor DropOut(Tensor t, int[] indices)
		{
			// only apply dropout when training
			if (!Need)
			{
				return t;
			}
			var res = new Tensor(t.Row, t.Col, Need);
			foreach (var index in indices)
			{
				res.W.Storage[index] = t.W.Storage[index];
			}
			_backwardOps.Add(delegate
			{
				foreach (var index in indices)
				{
					t.Grad.Storage[index] += res.Grad.Storage[index];
				}
			});
			return res;
		}


		public Tensor ElementwiseMultiply(Tensor l, Tensor r)
		{
			var res = new Tensor(l.Row, l.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = l.W.Storage[i] * r.W.Storage[i];
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					l.Grad.Storage[i] += r.W.Storage[i] * res.Grad.Storage[i];
					r.Grad.Storage[i] += l.W.Storage[i] * res.Grad.Storage[i];
				}
			});
			return res;
		}

		public Tensor HardSigmoid(Tensor t)
		{
			var res = new Tensor(t.Row, t.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				var val = t.W.Storage[i];
				var rval = 0f;
				if (val >= 2.5f)
				{
					rval = 1;
				}
				else if (val > -2.5f)
				{
					rval = 0.2f * val + 0.5f;
				}
				res.W.Storage[i] = rval;
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					var val = t.W.Storage[i];
					if (val < 2.5f && val > -2.5f)
					{
						t.Grad.Storage[i] += 0.2f * res.Grad.Storage[i];
					}
					else
					{
						t.Grad.Storage[i] += 0;
					}
				}
			});

			return res;
		}

		public Tensor HardTanh(Tensor t, float min, float max)
		{
			var res = new Tensor(t.Row, t.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				var val = t.W.Storage[i];
				var rval = min;
				if (val >= max)
				{
					rval = max;
				}
				else if (val > min)
				{
					rval = val;
				}
				res.W.Storage[i] = rval;
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < t.Grad.Storage.Length; i++)
				{
					var val = t.W.Storage[i];
					if (val < max && val > min)
					{
						t.Grad.Storage[i] += res.Grad.Storage[i];
					}
					else
					{
						t.Grad.Storage[i] += 0;
					}
				}
			});
			return res;
		}

		public Tensor Lookup(Tensor[] e, int id)
		{
			return e[id];
		}

		public Tensor Minus(float l, Tensor r)
		{
			var res = new Tensor(r.Row, r.Col, Need);
			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = l - r.W.Storage[i];
			}

			if (!Need)
			{
				return res;
			}
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < r.Grad.Storage.Length; i++)
				{
					r.Grad.Storage[i] -= res.Grad.Storage[i];
				}
			});
			return res;
		}


		public Tensor Multiply(Tensor l, Tensor r)
		{
			var nRow = l.Row;
			var nCol = r.Col;
			var nDot = l.Col;
            r.RefCount++;
			var res = new Tensor(nRow, nCol, Need);

			for (var i = 0; i < nRow; i++)
			{
				for (var k = 0; k < nDot; k++)
				{
					var fac = l.W[i, k];
					for (var j = 0; j < nCol; j++)
					{
						res.W[i, j] += fac * r.W[k, j];
					}
				}
			}
			// for (var i = 0; i < nRow; i++)
			// {
			// 	for (var j = 0; j < nCol; j++)
			// 	{
			// 		var dot = 0d;
			// 		for (var k = 0; k < nDot; k++)
			// 		{
			// 			dot += l.W[i, k] * r.W[k, j];
			// 		}
			// 		res.W[i, j] = dot;
			// 	}
			// }

			if (!Need) return res;

            if (l.Grad != null && r.Grad != null)
            {
                _backwardOps.Add(delegate
                {
                    for (var i = 0; i < nRow; i++)
                    {
                        for (var j = 0; j < nCol; j++)
                        {
                            for (var k = 0; k < nDot; k++)
                            {
                                l.Grad[i, k] += r.W[k, j] * res.Grad[i, j];
                                r.Grad[k, j] += l.W[i, k] * res.Grad[i, j];
                            }
                        }
                    }
                });
            }else if (l.Grad != null)
            {
                _backwardOps.Add(delegate
                {
                    for (var i = 0; i < nRow; i++)
                    {
                        for (var k = 0; k < nDot; k++)
                        {
                            for (var j = 0; j < nCol; j++)
                            {
                                l.Grad[i, k] += r.W[k, j] * res.Grad[i, j];
                            }
                        }
                    }
                });
            }else if (r.Grad != null)
            {
                _backwardOps.Add(delegate
                {
                    for (var i = 0; i < nRow; i++)
                    {
                        for (var k = 0; k < nDot; k++)
                        {
                            for(var j = 0; j < nCol; j++)
                            {
                                r.Grad[k, j] += l.W[i, k] * res.Grad[i, j];
                            }
                        }
                    }
                });
            }

			return res;
		}

		public Tensor Relu(Tensor t)
		{
			var res = new Tensor(t.Row, t.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = t.W.Storage[i] > 0 ? t.W.Storage[i] : 0;
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					t.Grad.Storage[i] += t.W.Storage[i] > 0 ? res.Grad.Storage[i] : 0;
				}
			});
			return res;
		}

		public Tensor Sigmoid(Tensor t)
		{
			var res = new Tensor(t.Row, t.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = 1.0f / (1f + (float)Math.Exp(-t.W.Storage[i]));
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					t.Grad.Storage[i] += (1 - res.W.Storage[i]) * res.W.Storage[i] * res.Grad.Storage[i];
				}
			});

			return res;
		}

		public Tensor Softmax(Tensor t)
		{
            var res = new Tensor(t.Row, t.Col, Need);

			// substract the maxium to obtain numerical stability
			// acutal softmax is softmax[i] = e^x[i] / sum_j(e^x[j])
            var maxInd = -1;
            for (var i = 0; i < t.W.Storage.Length; i++)
            {
                if (t.W.Storage[i] > t.W.Storage[maxInd])
                {
                    maxInd = i;
                }
            }
            var sum = 0f;
            var max = t.W.Storage[maxInd];
            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                    res.W.Storage[i] = (float)Math.Exp(t.W.Storage[i] - max);
                    sum += res.W.Storage[i];
            }
            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                res.W.Storage[i] /= sum;
            }

			if(!Need) return res;

            // numerical gradient is complicated,
			// direct compute w.r.t loss is prefered

			// gradient for output i w.r.t input k is 
			//     (1{i==k}o[k] - o[i]o[k])do[i]
			// gradient for input k is the sum of the aforementioned gradient
			// that is
			//     sum_i((1{i==k}o[k]-o[i]o[k])do[i])
			//     o[k]do[k] - sum_i(o[i]o[k]do[i])
			//     o[k]*(do[k] - sum_i(o[i]do[i])) is more compute efficient
            _backwardOps.Add(delegate
			{
                var dot = 0f;
                for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
                    dot += res.W.Storage[i] * res.Grad.Storage[i];
                }
				for (var i = 0; i < t.Grad.Storage.Length; i++)
				{
                    t.Grad.Storage[i] += res.W.Storage[i] * (res.Grad.Storage[i] - dot);
                }
            });

            return res;

        }

		public Tensor SoftmaxWithCrossEntropy(Tensor t, int[] mask, out float loss)
		{
            // for one sample
            // cross entropy loss or 
            // multinomainal logistic loss (special case if only one true label) is
            //     -sum_i(p(i)logq(i))
            //     -sum_k(1{y==k}logp(y==k|x;Theta))
			// goal is to minimize the loss

            // one true label, x is input, y is gold, o is softmax(x)
            //     - log o[y]
            // with softmax as activator
            //     log(sum_i(e^x[i])) - x[y]
			// gradient for x[i] is
			//     (1/sum_i(e^x[i]))*(e^x[i]) - 1{i==y}
			//     o[i] - 1{i==y}

            // for row vector only
            // value of mask is -1, 0, 1
            // mask should be interpreted as probability
            var res = new Tensor(t.Row, t.Col, Need);

			var maxInd = -1;
			for (var i = 0; i < t.W.Storage.Length; i++)
			{
				if (mask[i] >= 0 && (maxInd < 0 || t.W.Storage[i] > t.W.Storage[maxInd]))
				{
					maxInd = i;
				}
			}
			var sum = 0f;
			var max = t.W.Storage[maxInd];
			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				if (mask[i] >= 0)
				{
					res.W.Storage[i] = (float)Math.Exp(t.W.Storage[i] - max);
					sum += res.W.Storage[i];
				}
			}
			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] /= sum;
			}

			loss = 0;
			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				if (mask[i] > 0)
				{
					loss -= (float)Math.Log(res.W.Storage[i]) * mask[i];
				}
			}

			if (!Need) return res;


			_backwardOps.Add(delegate
			{
				for (var i = 0; i < t.Grad.Storage.Length; i++)
				{
					if (mask[i] >= 0)
					{
						t.Grad.Storage[i] += res.W.Storage[i] - mask[i];
					}
				}
			});

			return res;
		}



        public Tensor SoftmaxWithCrossEntropy(Tensor t, float[] target, out float loss)
        {
            // for one sample
            // cross entropy loss or 
            // multinomainal logistic loss is
            //     -sum_i(p(i)logq(i))
            // goal is to minimize the loss

            // with prob g, x is input, o is softmax(x)
            //     -sum_i(g[i] log o[i])
            // with softmax as activator
            //     sum_i(g[i](log(sum_k(e^x[k])) - x[i]))
            // gradient for x[j] is
            //     sum_i(g[i] ((1/sum_k(e^x[k]))*(e^x[j]) - 1{i==j}) )
            //     sum_i(g[i]((1/sum_k(e^x[k]))*(e^x[j]))) - sum_i(g[i] 1{i==j})
            //     sum_i(g[i]o[j]) - g[j]
            //     o[j] - g[j]


            // for row vector only
            // value of target is -1, 0 - 1, with sum(target)=1
            // target should be interpreted as probability
            var res = new Tensor(t.Row, t.Col, Need);

            var maxInd = -1;
            for (var i = 0; i < t.W.Storage.Length; i++)
            {
                if (target[i] >= 0 && (maxInd < 0 || t.W.Storage[i] > t.W.Storage[maxInd]))
                {
                    maxInd = i;
                }
            }
            var sum = 0f;
            var max = t.W.Storage[maxInd];
            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                if (target[i] >= 0)
                {
                    res.W.Storage[i] = (float)Math.Exp(t.W.Storage[i] - max);
                    sum += res.W.Storage[i];
                }
            }
            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                res.W.Storage[i] /= sum;
            }

            loss = 0f;
            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                if (target[i] > 0)
                {
                    loss -= (float)Math.Log(res.W.Storage[i]) * target[i];
                }
            }

            if (!Need) return res;


            _backwardOps.Add(delegate
            {
                for (var i = 0; i < t.Grad.Storage.Length; i++)
                {
                    if (target[i] >= 0)
                    {
                        t.Grad.Storage[i] += res.W.Storage[i] - target[i];
                    }
                }
            });

            return res;
        }

   //     public Tensor SoftmaxWithAverageRewardLoss(Tensor t, int[] mask, float advantage, out float loss)
   //     {
			//// objective function could be 
			////     expected immediate reward, 
			////     expected average reward, 
			////     expected discounted reward
			//// goal is to maximize the objective function
			////
			//// estimated gradient for a sample is
			////     average_t(gradent((log prob(at|st))Rt))
			//// x is input, o is softmax(x), y is the chosen, r is reward/advantage
			////     average_t(gradient((log o[y])r))

			//// loss could be the negate objective function
			//// goal it to minimize the loss

   //         // for row vector only
   //         // value of mask is -1, 0, 1
   //         // mask should be interpreted as probability
   //         var res = new Tensor(t.Row, t.Col, Need);

   //         var maxInd = -1;
   //         for (var i = 0; i < t.W.Storage.Length; i++)
   //         {
   //             if (mask[i] >= 0 && (maxInd < 0 || t.W.Storage[i] > t.W.Storage[maxInd]))
   //             {
   //                 maxInd = i;
   //             }
   //         }
   //         var sum = 0d;
   //         var max = t.W.Storage[maxInd];
   //         for (var i = 0; i < res.W.Storage.Length; i++)
   //         {
   //             if (mask[i] >= 0)
   //             {
   //                 res.W.Storage[i] = Math.Exp(t.W.Storage[i] - max);
   //                 sum += res.W.Storage[i];
   //             }
   //         }
   //         for (var i = 0; i < res.W.Storage.Length; i++)
   //         {
   //             res.W.Storage[i] /= sum;
   //         }

   //         loss = 0;
   //         for (var i = 0; i < res.W.Storage.Length; i++)
   //         {
   //             if (mask[i] > 0)
   //             {
			//		loss -= Math.Log(res.W.Storage[i] < 0 ? 1e-8 : res.W.Storage[i]) * advantage;
   //             }
   //         }

   //         if (!Need) return res;


   //         _backwardOps.Add(delegate
   //         {
   //             for (var i = 0; i < t.Grad.Storage.Length; i++)
   //             {
   //                 if (mask[i] >= 0)
   //                 {
   //                     t.Grad.Storage[i] += (res.W.Storage[i] - mask[i]) * advantage;
   //                 }
   //             }
   //         });

   //         return res;
   //     }


        public Tensor SoftmaxWithCrossEntropy(Tensor t, int gold, out float loss)
        {
            // for row vector only
            // value of mask is -1, 0, 1
            // mask should be interpreted as probability
            var res = new Tensor(t.Row, t.Col, Need);

            var maxInd = -1;
            for (var i = 0; i < t.W.Storage.Length; i++)
            {
                if (maxInd<0||t.W.Storage[i] > t.W.Storage[maxInd])
                {
                    maxInd = i;
                }
            }
            var sum = 0f;
            var max = t.W.Storage[maxInd];
            for (var i = 0; i < res.W.Storage.Length; i++)
            {

                res.W.Storage[i] = (float)Math.Exp(t.W.Storage[i] - max);
                sum += res.W.Storage[i];
            }
            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                res.W.Storage[i] /= sum;
            }

            loss = 0f;
            loss -= (float)Math.Log(res.W.Storage[gold]);

            if (!Need) return res;


            _backwardOps.Add(delegate
            {
                for (var i = 0; i < t.Grad.Storage.Length; i++)
                {
                    t.Grad.Storage[i] += res.W.Storage[i] - ((i == gold) ? 1 : 0);
                }
            });

            return res;
        }

	    public float Loss(Tensor x)
	    {
	        if (!Need) return x.W[0, 0];
	        _backwardOps.Add(delegate
	        {
	            x.Grad[0, 0] = 1;
	        });
            return x.W[0, 0];
	    }

        //public Tensor SoftmaxWithAveRewardLoss(Tensor t, int pred, float advantage, out float loss)
        //{
        //    // for row vector only
        //    // value of mask is -1, 0, 1
        //    // mask should be interpreted as probability
        //    var res = new Tensor(t.Row, t.Col, Need);

        //    var maxInd = -1;
        //    for (var i = 0; i < t.W.Storage.Length; i++)
        //    {
        //        if (maxInd < 0 || t.W.Storage[i] > t.W.Storage[maxInd])
        //        {
        //            maxInd = i;
        //        }
        //    }
        //    var sum = 0d;
        //    var max = t.W.Storage[maxInd];
        //    for (var i = 0; i < res.W.Storage.Length; i++)
        //    {

        //        res.W.Storage[i] = Math.Exp(t.W.Storage[i] - max);
        //        sum += res.W.Storage[i];
        //    }
        //    for (var i = 0; i < res.W.Storage.Length; i++)
        //    {
        //        res.W.Storage[i] /= sum;
        //    }

        //    loss = 0;
        //    loss -= res.W.Storage[pred] > 0 ? Math.Log(res.W.Storage[pred]) : 0;

        //    if (!Need) return res;


        //    _backwardOps.Add(delegate
        //    {
        //        for (var i = 0; i < t.Grad.Storage.Length; i++)
        //        {
        //            t.Grad.Storage[i] += res.W.Storage[i] - ((i == pred) ? advantage : 0);
        //        }
        //    });

        //    return res;
        //}

        public Tensor Tanh(Tensor t)
		{
			var res = new Tensor(t.Row, t.Col, Need);


			for (var i = 0; i < res.W.Storage.Length; i++)
			{
				res.W.Storage[i] = (float)Math.Tanh(t.W.Storage[i]);
			}

			if (!Need) return res;
			_backwardOps.Add(delegate
			{
				for (var i = 0; i < res.Grad.Storage.Length; i++)
				{
					t.Grad.Storage[i] += (1 - res.W.Storage[i] * res.W.Storage[i]) * res.Grad.Storage[i];
				}
			});
			return res;
		}
	}
}