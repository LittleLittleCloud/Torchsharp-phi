using TorchSharp;
using static TorchSharp.torch;

public class PhiLinear : nn.Module<Tensor, Tensor>
{
    protected Tensor? weight;
    protected readonly Tensor? bias;
    protected int inFeatures;
    protected int outFeatures;

    public PhiLinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(nameof(PhiLinear))
    {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        device = device ?? "cpu";
        this.weight = torch.randn(outFeatures, inFeatures, dtype: dtype, device: device);

        if (hasBias)
        {
            this.bias = torch.randn(outFeatures, dtype: dtype, device: device);
        }

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var dispose = torch.NewDisposeScope();

        // use float32
        var input2 = input.to_type(ScalarType.Float32);
        var weight2 = this.weight.to_type(ScalarType.Float32);
        var result = torch.matmul(input2, weight2.t());

        if (this.bias is not null)
        {
            result = result + this.bias.to_type(ScalarType.Float32);
        }

        result.Peek("result");
        return result.to_type(input.dtype).MoveToOuterDisposeScope();
    }
}
  