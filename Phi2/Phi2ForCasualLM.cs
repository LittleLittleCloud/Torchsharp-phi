using System.Text.Json;
using System.Text.Json.Serialization;
using FluentAssertions;
using Microsoft.ML.Tokenizers;
using ShellProgressBar;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

public class Phi2ForCasualLM
{
    private readonly PhiModelInferenceWrapper model;
    private readonly string device = "cpu";
    private readonly Phi2Tokenizer tokenizer;

    public Phi2ForCasualLM(PhiModelInferenceWrapper model, Phi2Tokenizer tokenizer, string device = "cpu")
    {
        this.model = model;
        this.device = device;
        this.tokenizer = tokenizer;
    }

    public PhiModelInferenceWrapper Model => this.model;

    public Phi2Tokenizer Tokenizer => this.tokenizer;

    public static Phi2ForCasualLM FromPretrained(
        string modelFolder,
        string configName = "config.json",
        string checkPointName = "phi-2.pt",
        ScalarType torchDtype = ScalarType.Float32,
        string device = "cpu")
    {
        var config = Path.Join(modelFolder, configName);
        var modelConfig = JsonSerializer.Deserialize<Phi2Config>(File.ReadAllText(config)) ?? throw new ArgumentNullException(nameof(config));
        modelConfig.Dtype = torchDtype;
        var phi = new Phi2Model(modelConfig);
        var wrapper = new PhiModelInferenceWrapper(phi);
        var loadedParameters = new Dictionary<string, bool>();
        wrapper.load_checkpoint(path: modelFolder, checkpointName: checkPointName, strict: true, loadedParameters: loadedParameters);
        wrapper = wrapper.to(device);
        wrapper.eval();
        var tokenzier = Phi2Tokenizer.FromPretrained(modelFolder);
        return new Phi2ForCasualLM(wrapper, tokenzier, device);
    }

    public string Device => this.device;

    public (
        Tensor, // output token ids [batch_size, sequence_length]
        Tensor // output logits [batch_size, sequence_length, vocab_size]
    ) Generate(
        Tensor inputIds, // input token ids [batch_size, sequence_length]
        Tensor attentionMask, // attention mask [batch_size, sequence_length]
        float temperature = 0.7f,
        float topP = 0.9f,
        int maxLen = 128,
        int[][]? stopTokenSequence = null,
        bool echo = false)
    {
        var batch = inputIds.shape[0];
        var device = inputIds.device;
        var minPromptLen = (int)inputIds.shape[1];
        var totalLen = minPromptLen + maxLen;
        if (stopTokenSequence == null)
        {
            stopTokenSequence = [[50256]];
        }
        else
        {
            // add 50265 to the stopTokenIds
            stopTokenSequence = stopTokenSequence.Append([50256]).Distinct().ToArray();
        }

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
                inputIds = torch.cat([inputIds, nextToken.unsqueeze(1)], dim: -1);
                attentionMask = torch.cat([attentionMask, attentionMask.new_ones(attentionMask.shape[0], 1)], dim: -1);
                foreach (var stopSequence in stopTokenSequence)
                {
                    // determine if the last n tokens are the stop sequence
                    var lastN = inputIds[.., ^stopSequence.Length..];
                    var lastNMatch = lastN == torch.tensor(stopSequence, device: device);
                    eosReached |= lastNMatch.all(dim: -1);
                }
                if (eosReached.all().item<bool>())
                {
                    // pBar.WriteLine("EOS reached");
                    // pBar.Tick(maxLen);
                    break;
                }

                var message = $"Generating Token {curPos}/{maxLen}";
                // pBar.Tick(curPos, message);
                var nextTokenIds = nextToken.to_type(ScalarType.Int32).data<int>().ToArray();
                var nextTokenStr = this.tokenizer.Decode(nextTokenIds);
                Console.Write(nextTokenStr);

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
