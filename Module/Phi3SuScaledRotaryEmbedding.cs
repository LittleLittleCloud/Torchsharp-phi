using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi;

public class Phi3SuScaledRotaryEmbedding: Phi3RotaryEmbedding
{
    private readonly double[] short_factor;
    private readonly double[] long_factor;
    private readonly int original_max_position_embeddings;
    private readonly int max_position_embeddings;
    private readonly double _base;

    public Phi3SuScaledRotaryEmbedding(int dim, Phi3Config config)
        : base(config.RopeTheta, config.MaxPositionEmbeddings, dim)
    {
        JsonElement shortFactorElement = (JsonElement)config.RopeScaling!["short_factor"];
        JsonElement longFactorDocument = (JsonElement)config.RopeScaling!["long_factor"];
        this.short_factor = shortFactorElement.EnumerateArray().Select(e => e.GetDouble()).ToArray();
        this.long_factor = longFactorDocument.EnumerateArray().Select(e => e.GetDouble()).ToArray();

        this.original_max_position_embeddings = config.OriginalMaxPositionEmbeddings;
        this.max_position_embeddings = config.MaxPositionEmbeddings;
        this._base = config.RopeTheta;
    }

    public override Phi3RotaryEmbeddingOutput forward(Phi3RotaryEmbeddingInput input)
    {
        var seq_len = (torch.max(input.PositionIds) + 1).ToInt32();
        var x = input.Input;
        Tensor ext_factors;
        if (seq_len > this.original_max_position_embeddings)
        {
            ext_factors = torch.tensor(this.long_factor, dtype: ScalarType.Float32, x.device);
        }
        else
        {
            ext_factors = torch.tensor(this.short_factor, dtype: ScalarType.Float32, x.device);
        }
        var inv_freq_shape = torch.arange(0, this.Dim, 2, dtype: ScalarType.Int64).to(torch.float32) / this.Dim;
        inv_freq_shape = inv_freq_shape.to(x.device);
        var inv_freq = 1.0f / (torch.pow(this._base, inv_freq_shape) * ext_factors);

        var inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(-1);
        inv_freq_expanded = inv_freq_expanded.expand(new long[] { input.PositionIds.shape[0], -1, 1 });
        var position_ids_expanded = input.PositionIds.unsqueeze(1).to(torch.float32);

        var freqs = inv_freq_expanded * position_ids_expanded;
        freqs = freqs.transpose(1, 2);
        var emb = torch.cat([freqs, freqs], dim: -1);
        var scale = (1.0 * this.max_position_embeddings) / this.original_max_position_embeddings;
        double scaling_factor;
        if (scale <= 1)
        {
            scaling_factor = 1.0;
        }
        else
        {
            scaling_factor = Math.Sqrt(1 + Math.Log(scale) / Math.Log(this.original_max_position_embeddings));
        }

        var cos = torch.cos(emb) * scaling_factor;
        var sin = torch.sin(emb) * scaling_factor;

        return new(cos.to_type(x.dtype), sin.to_type(x.dtype));
    }
}
