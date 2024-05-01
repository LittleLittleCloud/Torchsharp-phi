using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Phi;

public class Norm : Normalizer
{
    public override NormalizedString Normalize(string original)
    {
        // replace space with _
        var normalized = original.Replace(" ", "▁");

        return new NormalizedString(original, normalized, null, isOneToOneMapping: true);
    }
}

public class PreTokenizer : Microsoft.ML.Tokenizers.PreTokenizer
{
    public override IReadOnlyList<Split> PreTokenize(string sentence)
    {
        var split = new Split(sentence, new(0, sentence.Length));

        return new List<Split> { split };
    }
}

public class TokenizeDecoder : Microsoft.ML.Tokenizers.TokenizerDecoder
{
    private const char spaceReplacement = '▁';
    private string bos = "<s>";
    private string eos = "</s>";

    public TokenizeDecoder(string bos = "<s>", string eos = "</s>")
    {
        this.bos = bos;
        this.eos = eos;
    }

    public override string Decode(IEnumerable<string> tokens)
    {
        var str = string.Join("", tokens);
        str = str.Replace(spaceReplacement, ' ');

        if (str.StartsWith(bos))
        {
            str = str.Substring(bos.Length);
        }

        if (str.EndsWith(eos))
        {
            str = str.Substring(0, str.Length - eos.Length);
        }
        return str;
    }
}

public interface ITokenizer
{
    public int VocabSize { get; }

    public int PadId { get; }

    public int BosId { get; }

    public int EosId { get; }

    public string Decode(int[] input);

    public int[] Encode(string input, bool bos, bool eos);
}

/// <summary>
/// Copied from https://github.com/LittleLittleCloud/Torchsharp-llama/blob/main/ITokenizer.cs
/// </summary>
public class LLama2Tokenizer : ITokenizer
{
    private Tokenizer tokenizer;
    private bool addPrecedingSpace;

    public LLama2Tokenizer(string vocabPath, string mergesPath, bool addPrecedingSpace = true, int padToken = -1, int startToken = 1, int endToken = 2)
    {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;
        var bpe = new Bpe(vocabPath, mergesPath);
        this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new Norm());
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
        this.tokenizer.Decoder = decoder;
    }

    public LLama2Tokenizer(Dictionary<string, int> vocab, List<string> merges, bool addPrecedingSpace = true, int padToken = -1, int startToken = 1, int endToken = 2)
    {
        this.BosId = startToken;
        this.EosId = endToken;
        this.addPrecedingSpace = addPrecedingSpace;
        this.PadId = padToken;
        // save vocab to vocab-temp.json
        var vocabTempPath = "vocab-temp.json";
        var json = JsonSerializer.Serialize(vocab);
        File.WriteAllText(vocabTempPath, json);

        // save merges to merges-temp.txt
        var mergesTempPath = "merges-temp.txt";
        // filter out merges that contain newline character because it will cause error in BPE
        merges = merges.Where(x => !x.Contains('\r')).ToList();
        File.WriteAllLines(mergesTempPath, merges);

        var bpe = new Bpe(vocabTempPath, mergesTempPath);

        this.tokenizer = new Tokenizer(bpe, preTokenizer: new PreTokenizer(), normalizer: new Norm());
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
        this.tokenizer.Decoder = decoder;

        // delete temp files
        File.Delete(vocabTempPath);
        File.Delete(mergesTempPath);
    }

    public static LLama2Tokenizer FromPretrained(
        string folder,
        string tokenizerJsonPath = "tokenizer.json",
        string specialTokensMapPath = "special_tokens_map.json"
    )
    {
        tokenizerJsonPath = Path.Combine(folder, tokenizerJsonPath);
        var json = File.ReadAllText(tokenizerJsonPath);
        var jsonDocument = JsonDocument.Parse(json);
        // vocab: .model.vocab
        var vocabNode = jsonDocument.RootElement.GetProperty("model").GetProperty("vocab");

        // to Dictionary<string, int>
        var vocab = new Dictionary<string, int>();
        foreach (var item in vocabNode.EnumerateObject())
        {
            vocab[item.Name] = item.Value.GetInt32();
        }

        // added tokens: .added_tokens
        var addedTokensNode = jsonDocument.RootElement.GetProperty("added_tokens");
        foreach (var item in addedTokensNode.EnumerateArray())
        {
            // get id from item.id
            var id = item.GetProperty("id").GetInt32();
            var content = item.GetProperty("content").GetString()!;
            vocab[content] = id;
        }

        // merges: .model.merges
        var mergesNode = jsonDocument.RootElement.GetProperty("model").GetProperty("merges");
        // merges: List<string>
        var merges = new List<string>();
        foreach (var item in mergesNode.EnumerateArray())
        {
            merges.Add(item.GetString()!);
        }

        int startToken = 1, endToken = 2, padToken = -1;
        var specialTokenJsonPath = Path.Combine(folder, specialTokensMapPath);
        if (File.Exists(specialTokenJsonPath))
        {
            var specialTokenJson = File.ReadAllText(specialTokenJsonPath);
            var specialTokenMapDocument = JsonDocument.Parse(specialTokenJson);

            // retrieve bos_token, eos_token, pad_token if exists
            if (specialTokenMapDocument.RootElement.TryGetProperty("bos_token", out var bosTokenNode))
            {
                var bos_token_content = bosTokenNode.GetProperty("content").GetString()!;
                startToken = vocab[bos_token_content];
            }

            if (specialTokenMapDocument.RootElement.TryGetProperty("eos_token", out var eosTokenNode))
            {
                var eos_token_content = eosTokenNode.GetProperty("content").GetString()!;
                endToken = vocab[eos_token_content];
            }

            if (specialTokenMapDocument.RootElement.TryGetProperty("pad_token", out var padTokenNode))
            {
                var pad_token_content = padTokenNode.GetProperty("content").GetString()!;
                padToken = vocab[pad_token_content];
            }
        }

        return new LLama2Tokenizer(vocab, merges, padToken: padToken, addPrecedingSpace: false, startToken: startToken, endToken: endToken);
    }

    public int VocabSize => this.tokenizer.Model.GetVocabSize();

    public int PadId { get; }

    public int BosId { get; }

    public int EosId { get; }

    public string Decode(int[] input)
    {
        var str = this.tokenizer.Decode(input) ?? throw new Exception("Failed to decode");
        if (this.addPrecedingSpace)
        {
            str = str.TrimStart();
        }

        return str;
    }

    public int[] Encode(string input, bool bos, bool eos)
    {
        if (this.addPrecedingSpace)
        {
            input = " " + input;
        }
        var tokens = this.tokenizer.Encode(input).Ids.ToArray();
        if (bos)
        {
            tokens = new int[] { this.BosId }.Concat(tokens).ToArray();
        }
        if (eos)
        {
            tokens = tokens.Concat(new int[] { this.EosId }).ToArray();
        }

        Console.WriteLine($"tokens: {string.Join(",", tokens)}");

        return tokens;
    }
}