using TorchSharp;
using static TorchSharp.torch;

public class Phi2RotaryEmbedding : nn.Module<
    Tensor, // input
    int, // seq_len
    (
        Tensor, // cos
        Tensor // sin
    )>
{
    private readonly double _base;
    private readonly int _maxPositionEmbeddings;
    private readonly int _dim;

    public Phi2RotaryEmbedding(double baseValue, int maxPositionEmbeddings, int dim)
        : base(nameof(Phi2RotaryEmbedding))
    {
        _base = baseValue;
        _maxPositionEmbeddings = maxPositionEmbeddings;
        _dim = dim;
        var thetaNumerator = torch.arange(0, _dim, 2, dtype: ScalarType.Int64).to(torch.float32);
        this.register_buffer("inv_freq", torch.pow(baseValue, -1.0f * (thetaNumerator / dim)), persistent: false);
    }

    public int Dim => _dim;

    public override (Tensor, Tensor) forward(Tensor x, int seqLen)
    {
        // TODO
        // can be calculated once and cached
        var inv_freq = this.get_buffer("inv_freq").to(x.device);
        var t = torch.arange(seqLen, dtype: inv_freq.dtype, device: inv_freq.device);
        var freqs = torch.outer(t, inv_freq).to(torch.float32);
        var emb = torch.cat([freqs, freqs], dim: -1);

        var cos = torch.cos(emb);
        var sin = torch.sin(emb);

        return (cos[..seqLen].to_type(x.dtype), sin[..seqLen].to_type(x.dtype));
    }
}
  