using FluentAssertions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class Phi2Attention : nn.Module<
    Tensor, // hidden_states
    Tensor, // position_ids
    Tensor?, // attention_mask
    int, // past_key_value_length
    bool, // output_attentions
    (
        Tensor, // hidden_states,
        Tensor?, // attentions,
        Tensor? // present_key_value
    )>
{
    private readonly int? layerIdx;
    private Phi2Config config;
    private readonly double attentionDropout;
    private readonly int hiddenSize;
    private readonly int numAttentionHeads;
    private readonly int headDim;
    private readonly int numKeyValueHeads;
    private readonly int numKeyValueGroups;
    private readonly int maxPositionEmbeddings;
    private readonly double ropeTheta;
    private readonly double partialRotaryFactor;
    private readonly bool isCausal;
    private readonly PhiLinear q_proj;
    private readonly PhiLinear k_proj;
    private readonly PhiLinear v_proj;
    private readonly PhiLinear dense;
    private readonly bool qk_layernorm;
    private readonly LayerNorm? q_layernorm;
    private readonly LayerNorm? k_layernorm;

    private readonly Phi2RotaryEmbedding phiRotaryEmbedding;

    // cache_k, cache_v
    private Tensor cache_k;
    private Tensor cache_v;

    public Phi2Attention(Phi2Config config, int? layerIdx = null, int maxBatch = 2, int maxLength = 1024)
        : base(nameof(Phi2Attention))
    {
        this.layerIdx = layerIdx;
        this.config = config;
        
        this.attentionDropout = config.AttentionDropout;
        this.hiddenSize = config.HiddenSize;
        this.numAttentionHeads = config.NumAttentionHeads;
        this.headDim = this.hiddenSize / this.numAttentionHeads;
        this.numKeyValueHeads = config.NumKeyValueHeads ?? throw new ArgumentException("num_key_value_heads must be specified");
        this.numKeyValueGroups = this.numAttentionHeads / this.numKeyValueHeads;
        this.maxPositionEmbeddings = config.MaxPositionEmbeddings;
        this.ropeTheta = config.RopeTheta;
        this.partialRotaryFactor = config.PartialRotaryFactor;
        this.isCausal = true;

        (this.headDim * this.numAttentionHeads).Should().Be(this.hiddenSize, "hidden_size must be divisible by num_attention_heads");
        this.q_proj = new PhiLinear(this.hiddenSize, this.numAttentionHeads * this.headDim, hasBias: true, dtype: config.Dtype);
        this.k_proj = new PhiLinear(this.hiddenSize, this.numKeyValueHeads * this.headDim, hasBias: true, dtype: config.Dtype);
        this.v_proj = new PhiLinear(this.hiddenSize, this.numKeyValueHeads * this.headDim, hasBias: true, dtype: config.Dtype);
        this.dense = new PhiLinear(this.numAttentionHeads * this.headDim, this.hiddenSize, hasBias: true, dtype: config.Dtype);

        this.qk_layernorm = config.QkLayernorm;
        if (this.qk_layernorm)
        {
            this.q_layernorm = nn.LayerNorm(this.hiddenSize / this.numAttentionHeads, eps: config.LayerNormEps, elementwise_affine: true, dtype: config.Dtype);
            this.k_layernorm = nn.LayerNorm(this.hiddenSize / this.numAttentionHeads, eps: config.LayerNormEps, elementwise_affine: true, dtype: config.Dtype);
        }

        this.RegisterComponents();
        this.phiRotaryEmbedding = new Phi2RotaryEmbedding(
            dim: (int)(this.partialRotaryFactor * this.headDim),
            maxPositionEmbeddings: this.maxPositionEmbeddings,
            baseValue: this.config.RopeTheta);
        this.cache_k = torch.zeros(maxBatch, this.numKeyValueHeads, maxLength, this.headDim, dtype: config.Dtype);
        this.cache_v = torch.zeros(maxBatch, this.numKeyValueHeads, maxLength, this.headDim, dtype: config.Dtype);
    }

    public override (Tensor, Tensor?, Tensor?) forward(
        Tensor hiddenStates,
        Tensor positionIds,
        Tensor? attentionMask = null,
        int pastKeyValueLength = 0,
        bool outputAttentions = false)
    {
        // move cache to the same device as hiddenStates
        if (this.cache_k.device != hiddenStates.device)
        {
            this.cache_k = this.cache_k.to(hiddenStates.device, disposeAfter: true).DetachFromDisposeScope();
            this.cache_v = this.cache_v.to(hiddenStates.device, disposeAfter: true).DetachFromDisposeScope();
        }
        using var _ = torch.NewDisposeScope();
        var batchSize = (int)hiddenStates.shape[0];
        var seqLen = (int)hiddenStates.shape[1];

        var queryStates = this.q_proj.forward(hiddenStates);
        var keyStates = this.k_proj.forward(hiddenStates);
        var valueStates = this.v_proj.forward(hiddenStates);
        if (this.qk_layernorm)
        {
            queryStates = this.q_layernorm!.forward(queryStates);
            keyStates = this.k_layernorm!.forward(keyStates);
        }

        queryStates = queryStates.view(batchSize, seqLen, this.numAttentionHeads, this.headDim).transpose_(1, 2);
        keyStates = keyStates.view(batchSize, seqLen, this.numKeyValueHeads, this.headDim).transpose_(1, 2);
        valueStates = valueStates.view(batchSize, seqLen, this.numKeyValueHeads, this.headDim).transpose_(1, 2);
        var kvSeqLen = pastKeyValueLength == 0 ? (int)keyStates.shape[2] : pastKeyValueLength + (int)keyStates.shape[2];
        (var cos, var sin) = this.phiRotaryEmbedding.forward(valueStates, kvSeqLen);
        // split the last dim of queryStates and keyStates into rotary and non-rotary parts
        // shape: [batch_size, num_heads, seq_len, head_dim]
        // queryRot: [batch_size, num_heads, seq_len, :head_dim * partial_rotary_factor]
        // queryPass: [batch_size, num_heads, seq_len, head_dim * partial_rotary_factor:]
        
        var keyRot = keyStates[.., .., .., ..this.phiRotaryEmbedding.Dim];
        var keyPass = keyStates[.., .., .., this.phiRotaryEmbedding.Dim..];
        var queryRot = queryStates[.., .., .., ..this.phiRotaryEmbedding.Dim];
        var queryPass = queryStates[..,..,.., this.phiRotaryEmbedding.Dim..];
        (var qRot, var kRot) = Utils.ApplyRotaryPosEmb(queryRot, keyRot, cos, sin, positionIds);

        queryStates = torch.cat([qRot, queryPass], dim: -1);
        // update cache
        keyStates = torch.cat([kRot, keyPass], dim: -1);
        this.cache_k[..batchSize, .., pastKeyValueLength..kvSeqLen, ..] = keyStates;
        this.cache_v[..batchSize, .., pastKeyValueLength..kvSeqLen, ..] = valueStates;
        keyStates = this.cache_k[..batchSize, .., ..kvSeqLen, ..];
        valueStates = this.cache_v[..batchSize, .., ..kvSeqLen, ..];
        var keyStates2 = Utils.RepeatKV(keyStates, this.numKeyValueGroups).transpose(2, 3);
        var valueStates2 = Utils.RepeatKV(valueStates, this.numKeyValueGroups);
        // Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        var attnWeights = torch.matmul(queryStates.to_type(float32), keyStates2.to_type(float32));
        attnWeights = attnWeights / Math.Sqrt(this.headDim);
        if (attentionMask is not null)
        {
            attnWeights = attnWeights + attentionMask;
        }
        attnWeights = nn.functional.softmax(attnWeights, dim: -1);
        attnWeights = nn.functional.dropout(attnWeights, p: this.attentionDropout);
        var attnOutput = torch.matmul(attnWeights, valueStates2.to_type(float32)).to_type(hiddenStates.dtype);
        attnOutput = attnOutput.transpose_(1, 2).contiguous();
        attnOutput = attnOutput.reshape(batchSize, seqLen, this.hiddenSize);
        var result = this.dense.forward(attnOutput);
        return (result.MoveToOuterDisposeScope(), null, null);
    }
}
  