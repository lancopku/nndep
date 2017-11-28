using System;
using System.Text;

namespace nndep.Networks.Graph
{
    [Serializable]
    public class Matrix
	{
		public readonly int ColDim;
		public readonly int RowDim;
		public readonly float[] Storage;


		public Matrix(int rowDim, int colDim)
		{
			RowDim = rowDim;
			ColDim = colDim;
			Storage = new float[rowDim * colDim];
		}

		public float this[int row, int column]
		{
			get { return Storage[row * ColDim + column]; }
			set { Storage[row * ColDim + column] = value; }
		}


		public Matrix Copy()
		{
			var m = new Matrix(RowDim, ColDim);
			for (var i = 0; i < Storage.Length; ++i)
			{
				m.Storage[i] = Storage[i];
			}
			return m;
		}

		public void Fill(Func<float> valueFactory)
		{
			for (var i = 0; i < Storage.Length; i++)
			{
				Storage[i] = valueFactory();
			}
		}

		public void Fill(float[] arr)
		{
			for (var i = 0; i < Storage.Length; i++)
			{
				Storage[i] = arr[i];
			}
		}


		public int MaxIndex()
		{
			var mxv = Storage[0];
			var mxi = 0;
			for (var i = 1; i < Storage.Length; i++)
			{
				mxi = Storage[i] > mxv ? i : mxi;
				mxv = Storage[i] > mxv ? Storage[i] : mxv;
			}
			return mxi;
		}

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"[{RowDim}:{ColDim}]");
			foreach (var val in Storage)
			{
				sb.Append($" {val:##.0000}");
			}
			return sb.ToString();
		}
	}
}