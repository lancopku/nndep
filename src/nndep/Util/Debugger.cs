using System;
using System.Diagnostics;
using System.Linq;
using nndep.Networks.Graph;

namespace nndep.Util
{
	static class Checker
	{
		[Conditional("CHECK")]
		public static void Asserts<T>(bool condition, string message) where T : Exception, new()
		{
			if (!condition)
			{
				throw Activator.CreateInstance(typeof(T), message) as T;
			}
		}

		[Conditional("CHECK")]
		public static void Asserts<T>(bool condition) where T : Exception, new()
		{
			if (!condition)
			{
				throw new T();
			}
		}

		[Conditional("CHECK")]
		public static void Ensures<T>(bool condition, string message) where T : Exception, new()
		{
			if (!condition)
			{
				throw Activator.CreateInstance(typeof(T), message) as T;
			}
		}

		[Conditional("CHECK")]
		public static void Ensures<T>(bool condition) where T : Exception, new()
		{
			if (!condition)
			{
				throw new T();
			}
		}

		[Conditional("CHECK")]
		public static void IsColumnVector(this Matrix m)
		{
			if (m.ColDim != 1)
			{
				throw new ArgumentException("matrix is not a column vector", nameof(m));
			}
		}

		[Conditional("CHECK")]
		public static void IsColumnVector(this Tensor t)
		{
			if (t.Col != 1)
			{
				throw new ArgumentException("tensor is not a column vector", nameof(t));
			}
		}

		[Conditional("CHECK")]
		public static void IsDimEqual(this Tensor t, Tensor o)
		{
			if (t.Row != o.Row || t.Col != o.Col)
			{
				throw new ArgumentException("tensor dim misaligned", nameof(o));
			}
		}

		[Conditional("CHECK")]
		public static void IsMulCompatible(this Tensor l, Tensor r)
		{
			if (l.Col != r.Row)
			{
				throw new ArgumentException("matmul tensor dim misaligned", nameof(r));
			}
		}

		[Conditional("CHECK")]
		public static void IsMulResult(this Tensor t, Tensor l, Tensor r)
		{
			if (t.Row != l.Row || t.Col != r.Col)
			{
				throw new ArithmeticException("matmul implementation broken, result dim is wrong");
			}
		}

		[Conditional("CHECK")]
		[Conditional("TUNE")]
		public static void IsNan(this Matrix m)
		{
			if (m.Storage.Any(float.IsNaN))
			{
				throw new ArithmeticException("not a number");
			}
		}

		[Conditional("CHECK")]
		[Conditional("TUNE")]
		public static void IsNan(this Matrix[] m)
		{
			if (m.Any(mat => mat.Storage.Any(float.IsNaN)))
			{
				throw new ArithmeticException("not a number");
			}
		}

		[Conditional("CHECK")]
		public static void IsRowRangeBounded(this Tensor t, int[] inds)
		{
			if (!inds.All(ind => ind < t.Row))
			{
				throw new ArgumentException("indices out of range", nameof(inds));
			}
		}

		[Conditional("CHECK")]
		public static void Requires<T>(bool condition, string message) where T : Exception, new()
		{
			if (!condition)
			{
				throw Activator.CreateInstance(typeof(T), message) as T;
			}
		}

		[Conditional("CHECK")]
		public static void Requires<T>(bool condition) where T : Exception, new()
		{
			if (!condition)
			{
				throw new T();
			}
		}
	}
}