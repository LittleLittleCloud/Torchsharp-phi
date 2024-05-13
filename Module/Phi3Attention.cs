using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Phi;

public class Phi3AttentionInput
{
    public Phi3AttentionInput(
        Tensor hidden_states,
        Tensor position_ids,
        Tensor? attention_mask = null,
        IKVCache? cache = null,
        bool output_attentions = false)
    {
        this.hidden_states = hidden_states;
        this.attention_mask = attention_mask;
        this.position_ids = position_ids;
        this.cache = cache;
        this.output_attentions = output_attentions;
    }
    public Tensor hidden_states { get; set; }

    public Tensor? attention_mask { get; set; }

    public Tensor position_ids { get; set; }

    public IKVCache? cache { get; set; }

    public bool output_attentions { get; set; } = false;
}

public class Phi3AttentionOutput
{
    public Phi3AttentionOutput(
        Tensor hidden_states,
        Tensor? attentions = null,
        IKVCache? cache = null)
    {
        this.hidden_states = hidden_states;
        this.attentions = attentions;
        this.cache = cache;
    }

    public Tensor hidden_states { get; set; }

    public Tensor? attentions { get; set; }

    public IKVCache? cache { get; set; }
}

public class Phi3Attention : nn.Module<Phi3AttentionInput, Phi3AttentionOutput>
{
    private readonly Phi3Config config;
    private readonly int layer_idx;
    private readonly double attention_dropout;
    private readonly int hidden_size;
    private readonly int num_heads;
    private readonly int head_dim;
    private readonly int num_key_value_heads;
    private readonly int num_key_value_groups;
    private readonly int max_position_embeddings;
    private readonly int original_max_position_embeddings;
    private readonly double rope_theta;
    private readonly Dictionary<string, object>? rope_scaling;
    private readonly bool is_causal;

    private readonly Linear o_proj;
    private readonly Linear qkv_proj;
    private nn.Module<Phi3RotaryEmbeddingInput, Phi3RotaryEmbeddingOutput> rotary_emb;

    public Phi3Attention(Phi3Config config, int layer_idx)
        : base(nameof(Phi3Attention))
    {
        this.config = config;
        this.layer_idx = layer_idx;
        this.attention_dropout = config.AttentionDropout;
        this.hidden_size = config.HiddenSize;
        this.num_heads = config.NumAttentionHeads;
        this.head_dim = this.hidden_size / this.num_heads;
        this.num_key_value_heads = config.NumKeyValueHeads ?? throw new ArgumentException("num_key_value_heads must be specified");
        this.num_key_value_groups = this.num_heads / this.num_key_value_heads;
        this.max_position_embeddings = config.MaxPositionEmbeddings;
        this.original_max_position_embeddings = config.OriginalMaxPositionEmbeddings;
        this.rope_theta = config.RopeTheta;
        this.rope_scaling = config.RopeScaling;
        this.is_causal = true;

        (this.head_dim * this.num_heads).Should().Be(this.hidden_size, "hidden_size must be divisible by num_heads");

        var op_size = this.num_heads * this.head_dim + 2 * (this.num_key_value_heads * this.head_dim);
        this.o_proj = nn.Linear(this.num_heads * this.head_dim, this.hidden_size, hasBias: false, dtype: config.DType);
        this.qkv_proj = nn.Linear(this.hidden_size, op_size, hasBias: false, dtype: config.DType);
        this._init_rope();
    }

    private void _init_rope()
    {
        if (this.rope_scaling is null)
        {
            this.rotary_emb = new Phi3RotaryEmbedding(this.rope_theta, this.max_position_embeddings, this.head_dim);
        }
        else
        {
            this.rotary_emb = new Phi3SuScaledRotaryEmbedding(this.head_dim, this.config);
        }
    }

    public override Phi3AttentionOutput forward(Phi3AttentionInput input)
    {
        using (var _ = NewDisposeScope())
        {
            var hidden_states = input.hidden_states;
            var positionIds = input.position_ids;
            var output_attentions = input.output_attentions;
            var bsz = hidden_states.shape[0];
            var q_len = hidden_states.shape[1];

            var qkv = this.qkv_proj.forward(hidden_states);
            var query_pos = this.num_heads * this.head_dim;
            var query_states = qkv[.., .., ..query_pos];
            var key_states = qkv[.., .., query_pos..(query_pos + this.num_key_value_heads * this.head_dim)];
            var value_states = qkv[.., .., (query_pos + this.num_key_value_heads * this.head_dim)..];
            query_states = query_states.view(bsz, q_len, this.num_heads, this.head_dim).transpose(1, 2);
            key_states = key_states.view(bsz, q_len, this.num_key_value_heads, this.head_dim).transpose(1, 2);
            value_states = value_states.view(bsz, q_len, this.num_key_value_heads, this.head_dim).transpose(1, 2);

            var kv_seq_len = key_states.IntShape()[^2];
            var past_key_value = input.cache;
            if (past_key_value is not null)
            {
                kv_seq_len += past_key_value.GetUsableLength(kv_seq_len, this.layer_idx);
            }

            var embOutput = this.rotary_emb.forward(new Phi3RotaryEmbeddingInput(value_states, positionIds, kv_seq_len));
            (var cos, var sin) = (embOutput.Cos, embOutput.Sin);

            (query_states, key_states) = Utils.ApplyRotaryPosEmb(query_states, key_states, cos, sin);

            if (past_key_value is not null)
            {
                (key_states, value_states) = past_key_value.UpdateKVCache(key_states, value_states, this.layer_idx);
            }

            // repeat k/v heads if n_kv_heads < n_heads
            key_states = Utils.Phi3RepeatKV(key_states, this.num_key_value_groups);
            value_states = Utils.Phi3RepeatKV(value_states, this.num_key_value_groups);

            var attn_weights = torch.matmul(query_states, key_states.transpose(2, 3));
            attn_weights = attn_weights / Math.Sqrt(this.head_dim);

            attn_weights.shape.Should().BeEquivalentTo(new long[] { bsz, this.num_heads, q_len, kv_seq_len });

            var attention_mask = input.attention_mask;
            if (attention_mask is not null)
            {
                attention_mask.shape.Should().BeEquivalentTo(new long[] { bsz, 1, q_len, kv_seq_len });
                attn_weights = attn_weights + attention_mask;
            }

            // upscale attention to fp32 to avoid overflow
            attn_weights = nn.functional.softmax(attn_weights, dim: -1, dtype: ScalarType.Float32).to(value_states.dtype);
            attn_weights = nn.functional.dropout(attn_weights, this.attention_dropout, this.training);

            var attn_output = torch.matmul(attn_weights, value_states);

            attn_output.shape.Should().BeEquivalentTo(new long[] { bsz, this.num_heads, q_len, this.head_dim });

            attn_output = attn_output.transpose(1, 2).contiguous();
            attn_output = attn_output.reshape(bsz, q_len, this.hidden_size);

            attn_output = this.o_proj.forward(attn_output);

            return new(attn_output.MoveToOuterDisposeScope(), output_attentions ? attn_weights.MoveToOuterDisposeScope() : null, past_key_value);
        }
    }
}
