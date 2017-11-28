using System;
using System.Runtime.Serialization;
using System.Text;

namespace nndep.Networks.Graph
{
    [Serializable]
    public class Tensor
	{
        [NonSerialized] public Matrix Grad;
		public readonly Matrix W;
		public Matrix Mom;
		public Matrix VarMom;
        public int RefCount;

		public Tensor(int rowDimension, int columnDimension, bool grad)
		{
			W = new Matrix(rowDimension, columnDimension);
			Grad = grad ? new Matrix(rowDimension, columnDimension) : null;
			Mom = null;
			VarMom = null;
            RefCount = 0;
		}

		public int Row => W.RowDim;
		public int Col => W.ColDim;

		public void Fill(Func<float> valueFactory)
		{
			W.Fill(valueFactory);
		}

		public void Fill(float[] val)
		{
			W.Fill(val);
		}

        //public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        //{
        //    info.AddValue(nameof(W), W);
        //    info.AddValue(nameof(Mom), Mom);
        //    info.AddValue(nameof(VarMom), VarMom);
        //}

        //protected Tensor(SerializationInfo info, StreamingContext context)
        //{
        //    W = (Matrix)info.GetValue(nameof(W), typeof(Matrix));
        //    Mom = (Matrix)info.GetValue(nameof(Mom), typeof(Matrix));
        //    VarMom = (Matrix)info.GetValue(nameof(VarMom), typeof(Matrix));
        //    Grad = new Matrix(Row, Col);
        //}

        [OnDeserialized()]
        internal void OnDeserializedMethod(StreamingContext context)
        {
            Grad = new Matrix(Row, Col);
        }

        public void InitParam(bool mom, bool varmom)
		{
			if (mom)
			{
				Mom = new Matrix(Row, Col);
			}
			if (varmom)
			{
				VarMom = new Matrix(Row, Col);
			}
		}

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"[{Row}:{Col}]");
			for (var i = 0; i < Row; ++i)
			{
				for (var j = 0; j < Col; ++j)
				{
					sb.Append($" {W[i, j]:f4}");
				}
				sb.AppendLine();
			}
			return sb.ToString();
		}
	}
}