using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi.Pipeline;

public class CasualLMPipeline
{
    private readonly ITokenizer tokenizer;
    private readonly nn.Module<CasualLMModelInput, CasualLMModelOutput> model;

    public CasualLMPipeline(
        ITokenizer tokenizer,
        nn.Module<CasualLMModelInput, CasualLMModelOutput> model,
        string device = "cpu")
    {
        this.tokenizer = tokenizer;
        this.model = model;
        this.Device = device;
    }

    public ITokenizer Tokenizer => this.tokenizer;

    public nn.Module<CasualLMModelInput, CasualLMModelOutput> Model => this.model;

    public Device Device { get; }

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
            stopTokenSequence = [[this.tokenizer.EosId]];
        }
        else
        {
            stopTokenSequence = stopTokenSequence.Append([this.tokenizer.EosId]).Distinct().ToArray();
        }

        using (var _ = torch.no_grad())
        {
            var prevPos = 0;
            var eosReached = torch.tensor(new bool[batch], device: device);
            torch.Tensor? logits = default;
            var cache = new DynamicKVCache();
            if (minPromptLen == totalLen)
            {
                var input = new CasualLMModelInput(inputIds, attentionMask, past_key_values_length: 0);
                var output = this.model.forward(input);
                logits = output.logits;
            }
            for (int curPos = minPromptLen; curPos != totalLen; curPos++)
            {
                var input = new CasualLMModelInput(inputIds[.., prevPos..curPos], attentionMask[.., prevPos..curPos], past_key_values_length: prevPos);
                var output = this.model.forward(input);
                logits = output.logits;
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
