﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using static TorchSharp.torch;

public class Phi3Config
{
    //vocab_size=32064,
    //hidden_size=3072,
    //intermediate_size=8192,
    //num_hidden_layers=32,
    //num_attention_heads=32,
    //num_key_value_heads=None,
    //resid_pdrop=0.0,
    //embd_pdrop=0.0,
    //attention_dropout=0.0,
    //hidden_act="silu",
    //max_position_embeddings=4096,
    //original_max_position_embeddings=4096,
    //initializer_range=0.02,
    //rms_norm_eps=1e-5,
    //use_cache=True,
    //tie_word_embeddings=False,
    //rope_theta=10000.0,
    //rope_scaling=None,
    //bos_token_id=1,
    //eos_token_id=32000,
    //pad_token_id=32000,
    //sliding_window=None,
    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = 32064;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 3072;

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 8192;

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; } = 32;

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; } = 32;

    [JsonPropertyName("num_key_value_heads")]
    public int? NumKeyValueHeads { get; set; } = null;

    [JsonPropertyName("resid_pdrop")]
    public double ResidPdrop { get; set; } = 0.0;

    [JsonPropertyName("embd_pdrop")]
    public double EmbdPdrop { get; set; } = 0.0;

    [JsonPropertyName("attention_dropout")]
    public double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "silu";

    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; set; } = 4096;

    [JsonPropertyName("original_max_position_embeddings")]
    public int OriginalMaxPositionEmbeddings { get; set; } = 4096;

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; } = 0.02;

    [JsonPropertyName("rms_norm_eps")]
    public double LayerNormEps { get; set; } = 1e-5;

    [JsonPropertyName("use_cache")]
    public bool UseCache { get; set; } = true;

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; } = false;

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; } = 10000.0;

    [JsonPropertyName("rope_scaling")]
    public double? RopeScaling { get; set; } = null;

    [JsonPropertyName("partial_rotary_factor")]
    public double PartialRotaryFactor { get; set; } = 0.5;

    [JsonPropertyName("qk_layernorm")]
    public bool QkLayernorm { get; set; } = false;

    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; set; } = 1;

    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; set; } = 32000;

    [JsonPropertyName("pad_token_id")]
    public int PadTokenId { get; set; } = 32000;

    [JsonPropertyName("sliding_window")]
    public int? SlidingWindow { get; set; } = null;

    public ScalarType DType { get; set; } = ScalarType.BFloat16;
}