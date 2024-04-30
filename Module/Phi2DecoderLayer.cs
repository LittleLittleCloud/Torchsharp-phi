using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class Phi2DecoderLayer : nn.Module<
    Tensor, // hidden_states
    Tensor, // position_ids
    Tensor?, // attention_mask
    int, // past_key_value_length
    bool, // use_cache
    bool, // output_attentions
    (
        Tensor, // hidden_states,
        Tensor?, // attentions,
        Tensor? // present_key_value
    )>
{
    private readonly int? layerIdx;
    private Phi2Attention self_attn;
    private Phi2MLP mlp;
    private LayerNorm input_layernorm;
    private Dropout resid_dropout;

    public Phi2DecoderLayer(Phi2Config config, int? layerIdx = null)
        : base(nameof(Phi2DecoderLayer))
    {
        this.layerIdx = layerIdx;
        this.self_attn = new Phi2Attention(config, layerIdx);
        this.mlp = new Phi2MLP(config);
        this.input_layernorm = nn.LayerNorm(config.HiddenSize, eps: config.LayerNormEps, dtype: config.Dtype);
        this.resid_dropout = nn.Dropout(config.ResidPdrop);
    }

    public override (Tensor, Tensor?, Tensor?) forward(
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor? attentionMask = null,
        int pastKeyValueLength = 0,
        bool useCache = false,
        bool outputAttentions = false)
    {
        using var _ = torch.NewDisposeScope();
        var residual = hiddenStates;
        hiddenStates = this.input_layernorm.forward(hiddenStates);
        (var attnOutput, var attnWeights, var presentKeyValue) = this.self_attn.forward(
            hiddenStates: hiddenStates,
            positionIds: positionIds,
            attentionMask: attentionMask,
            pastKeyValueLength: pastKeyValueLength,
            outputAttentions: outputAttentions);
        var feed_forward_hiddenStates = this.mlp.forward(hiddenStates);
        hiddenStates = residual + feed_forward_hiddenStates + attnOutput;

        return (hiddenStates.MoveToOuterDisposeScope(), null, null);
    }
}
  