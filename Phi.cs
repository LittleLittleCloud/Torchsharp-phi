using System.Text.Json;
using System.Text.Json.Serialization;
using FluentAssertions;
using Microsoft.ML.Tokenizers;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

public class PhiForCasualLM
{
    private readonly PhiModelInferenceWrapper model;

    public PhiForCasualLM(PhiModelInferenceWrapper model)
    {
        this.model = model;
    }

    public PhiModelInferenceWrapper Model => this.model;

    public static PhiForCasualLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string weightsName = "phi-2.pt",
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<PhiConfig>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        var phi = new PhiModel(modelConfig);
        var wrapper = new PhiModelInferenceWrapper(phi);
        var weightPath = Path.Join(modelFolder, weightsName);
        var loadedParameters = new Dictionary<string, bool>();
        wrapper.load_py(weightPath, strict: true, loadedParameters: loadedParameters);
        
        wrapper = wrapper.to(device);

        return new PhiForCasualLM(wrapper);
    }

    // public (
    //     Tensor, // output token ids [batch_size, sequence_length]
    //     Tensor // output logits [batch_size, sequence_length, vocab_size]
    // ) Generate(
    //     Tensor inputIds, // input token ids [batch_size, sequence_length]
    //     Tensor attentionMask, // attention mask [batch_size, sequence_length]
    //     float temperature = 0.7f,
    //     float topP = 0.9f,
    //     int maxLen = 128,
    //     bool echo = false)
    // {
    //     var batch = inputIds.shape[0];
    //     var device = inputIds.device;

    //     using (var _ = torch.no_grad())
    //     {
    //         var prevPos = 0;
    //         var eosReached = torch.tensor(new bool[batch], device: device);
    //         torch.Tensor logits;
    //         if (minPromptLen == totalLen)
    //         {
    //             logits = this.transformer.forward(tokens, prevPos);
    //             tokenLogProbs = -torch.nn.functional.cross_entropy(input: logits.transpose(1, 2), target: tokens, reduction: torch.nn.Reduction.None, ignore_index: this.tokenizer.PadId);
    //         }

    //         for (int curPos = minPromptLen; curPos != totalLen; curPos++)
    //         {
    //             logits = this.transformer.forward(tokens[.., prevPos..curPos], prevPos);
    //             torch.Tensor nextToken;
    //             if (temperature > 0)
    //             {
    //                 var probs = torch.softmax(logits[.., -1] / temperature, dim: -1);
    //                 nextToken = this.SampleTopP(probs, topP);
    //             }
    //             else
    //             {
    //                 nextToken = torch.argmax(logits[.., -1], dim: -1);
    //             }

    //             nextToken = nextToken.reshape(-1);
    //             // # only replace token if prompt has already been generated
    //             nextToken = torch.where(attentionMask[.., curPos], tokens[.., curPos], nextToken);

    //             // print nextToken
    //             Console.WriteLine($"nextToken: {string.Join(",", nextToken.data<long>())}");

    //             // print curPos
    //             Console.WriteLine($"curPos: {curPos}");
    //             tokens[.., curPos] = nextToken;
    //             if (logProbs)
    //             {
    //                 tokenLogProbs![.., (prevPos + 1) .. (curPos + 1)] = - torch.nn.functional.cross_entropy(input: logits.transpose(1, 2), target: tokens[.., (prevPos + 1) .. (curPos + 1)], reduction: torch.nn.Reduction.None, ignore_index: this.tokenizer.PadId);
    //             }

    //             eosReached |= (~attentionMask[.., curPos]) & (nextToken == this.tokenizer.EosId);
    //             if (eosReached.all().item<bool>())
    //             {
    //                 break;
    //             }

    //             prevPos = curPos;
    //         }

    //         var outputTokens = new int[batch][];
    //         var outputLogProbs = new float[batch][];

    //         for (var i = 0; i < batch; i++)
    //         {
    //             // cut to max gen len
    //             var start = echo ? 0 : promptTokens[i].Length;
    //             var toks = tokens[i][start..(promptTokens[i].Length + maxGenLen)].data<long>().Select(x => (int)x).ToArray();
    //             float[]? probs = null;
    //             if (logProbs)
    //             {
    //                 probs = tokenLogProbs![i][start..(promptTokens[i].Length + maxGenLen)].data<float>().ToArray();
    //             }

    //             // cut to first eos if any
    //             if (toks.Contains(this.tokenizer.EosId))
    //             {
    //                 var eosPos = Array.IndexOf(toks, this.tokenizer.EosId);
    //                 toks = toks[..eosPos];
    //                 if (logProbs)
    //                 {
    //                     probs = probs![..eosPos];
    //                 }
    //             }

    //             outputTokens[i] = toks;
    //             if (logProbs)
    //             {
    //                 outputLogProbs[i] = probs!;
    //             }
    //         }

    //         return (outputTokens, logProbs ? null : outputLogProbs);
    //     }
    // }

    private torch.Tensor SampleTopP(torch.Tensor logits, float topP)
    {
        (var probsSort, var probsIndex) = torch.sort(logits, dim: -1, descending: true);
        var cumsum = torch.cumsum(probsSort, dim: -1);
        var mask = cumsum - probsSort > topP;
        probsSort[mask] = 0f;
        probsSort /= probsSort.sum(dim: -1, keepdim: true);
        var nextToken = torch.multinomial(probsSort, num_samples: 1);
        nextToken = torch.gather(probsIndex, dim: -1, index: nextToken);
        return nextToken;
    }
}