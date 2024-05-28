using System.Reflection.PortableExecutable;
using System.Text.Json;
using Microsoft.ML.Tokenizers;
using Phi;

public class Phi2Tokenizer : ITokenizer
{
    private CodeGen tokenizer;
    private bool addPrecedingSpace;
    private readonly int _bosId;
    private readonly int _eosId;

    public Phi2Tokenizer(
        CodeGen tokenizer,
        bool addPrecedingSpace,
        int bosId,
        int eosId)
    {
        this.tokenizer = tokenizer;
        this.addPrecedingSpace = addPrecedingSpace;
        _bosId = bosId;
        _eosId = eosId;
    }

    public static Phi2Tokenizer FromPretrained(
        string folder,
        string vocabFile = "vocab.json",
        string mergesFile = "merges.txt",
        string specialTokensFile = "special_tokens_map.json",
        bool addPrecedingSpace = false)
    {
        var vocabPath = Path.Combine(folder, vocabFile);
        var mergesPath = Path.Combine(folder, mergesFile);
        var specialTokenMapPath = Path.Combine(folder, specialTokensFile);
        using var vocabStream = File.OpenRead(vocabPath);
        using var mergesStream = File.OpenRead(mergesPath);
        var codeGenTokenizer = Tokenizer.CreatePhi2(vocabStream, mergesStream) as CodeGen;
        return new Phi2Tokenizer(codeGenTokenizer!, addPrecedingSpace, codeGenTokenizer!.BeginningOfSentenceId.GetValueOrDefault(), codeGenTokenizer.EndOfSentenceId.GetValueOrDefault());
    }

    public int BosId { get => this.tokenizer.BeginningOfSentenceId!.Value; }

    public int EosId { get => this.tokenizer.EndOfSentenceId!.Value; }

    public string Decode(int[] input)
    {
        var str = this.tokenizer.Decode(input) ?? throw new Exception("Failed to decode");
        if (this.addPrecedingSpace)
        {
            str = str.TrimStart();
        }

        return str;
    }


    public int[] Encode(string input, bool bos = false, bool eos = false)
    {
        if (this.addPrecedingSpace)
        {
            input = " " + input;
        }
        var tokens = this.tokenizer.EncodeToIds(input).ToArray();
        if (bos)
        {
            tokens = new int[] { _bosId }.Concat(tokens).ToArray();
        }
        if (eos)
        {
            tokens = tokens.Concat(new int[] { _eosId }).ToArray();
        }
        return tokens;
    }
}
