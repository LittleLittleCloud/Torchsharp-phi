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
        string weightsName = "phi-2-float32.pt",
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

    public (
        Tensor, // output token ids [batch_size, sequence_length]
        Tensor // output logits [batch_size, sequence_length, vocab_size]
    ) Generate(
        Tensor inputIds, // input token ids [batch_size, sequence_length]
        Tensor attentionMask, // attention mask [batch_size, sequence_length]
        float temperature = 0.7f,
        float topP = 0.9f,
        int maxLen = 128,
        int stopTokenId = 50256,
        bool echo = false)
    {
        var batch = inputIds.shape[0];
        var device = inputIds.device;
        var minPromptLen = (int)inputIds.shape[1];
        var totalLen = minPromptLen + maxLen;

        using (var _ = torch.no_grad())
        {
            var prevPos = 0;
            var eosReached = torch.tensor(new bool[batch], device: device);
            torch.Tensor? logits = default;
            if (minPromptLen == totalLen)
            {
                (logits, var _, var _) = this.model.forward(inputIds, attentionMask, prevPos);
            }

            for (int curPos = minPromptLen; curPos != totalLen; curPos++)
            {
                (logits, var _, var _) = this.model.forward(inputIds[.., prevPos..curPos], attentionMask[.., prevPos..curPos], prevPos);
                torch.Tensor nextToken;
                if (temperature > 0)
                {
                    var probs = torch.softmax(logits[.., -1] / temperature, dim: -1);
                    nextToken = this.SampleTopP(probs, topP);
                }
                else
                {
                    nextToken = torch.argmax(logits[.., -1], dim: -1);
                }

                nextToken = nextToken.reshape(-1);
                nextToken.Peek("nextToken");
                // nextToken = torch.where(attentionMask[.., curPos], inputIds[.., curPos], nextToken);

                // print nextToken
                Console.WriteLine($"nextToken: {string.Join(",", nextToken.data<long>())}");

                // print curPos
                Console.WriteLine($"curPos: {curPos}");
                inputIds = torch.cat([inputIds, nextToken.unsqueeze(1)], dim: -1);
                attentionMask = torch.cat([attentionMask, attentionMask.new_ones(attentionMask.shape[0], 1)], dim: -1);

                // eosReached |= (~attentionMask[.., curPos]) & (nextToken == stopTokenId);
                // if (eosReached.all().item<bool>())
                // {
                //     break;
                // }

                prevPos = curPos;

            }

            return (inputIds, logits!);
        }
    }

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