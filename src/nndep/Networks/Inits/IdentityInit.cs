namespace nndep.Networks.Inits
{
	class IdentityInit : IInit
	{
		private readonly float _val;

		public IdentityInit(float val)
		{
			_val = val;
		}

		public float Next()
		{
			return _val;
		}

		public override string ToString()
		{
			return $"IdentityInit[{_val}]";
		}
	}
}