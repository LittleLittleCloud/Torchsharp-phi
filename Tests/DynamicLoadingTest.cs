using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Phi.Module;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace Phi.Tests;

public class DynamicLoadingTest
{
    private ITestOutputHelper output;
    private int testDimension = 1024 * 2 * 2;
    private int preHeat = 1;
    private int loops = 10;
    private int numLayers = 512;
    public DynamicLoadingTest(ITestOutputHelper output)
    {
        this.output = output;
    }

    [Fact]
    public async Task ItGetSizeInBytesTestAsync()
    {
        // meta is critical for the test
        // as the size of the model to test is 372 GB
        // and can't be loaded in real device like cpu or cuda
        var device = "meta";
        var model = new PhiLinear(100_000, 1000_000, device: device);

        var sizeInBytes = model.GetSizeInBytes();

        var sizeInGigaBytes = sizeInBytes / 1024 / 1024 / 1024;
        sizeInGigaBytes.Should().Be(372);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ItInferDeviceMapTestsAsync()
    {
        var device = "meta";
        var model = new SequentialLinear(40_000, device);

        // cuda:0 has 24 GB
        // cpu has 20 GB
        // disk has 2 TB
        var deviceMap = model.InferDeviceMapForEachLayer(
            devices: ["cuda:0", "cpu", "disk"],
            deviceSizeMapInByte: new Dictionary<string, long>
            {
                ["cuda:0"] = 24L * 1024 * 1024 * 1024,
                ["cpu"] = 20L * 1024 * 1024 * 1024,
                ["disk"] = 2L * 1024 * 1024 * 1024 * 1024,
            });

        var json = JsonSerializer.Serialize(deviceMap, new JsonSerializerOptions { WriteIndented = true });
        Approvals.Verify(json);
    }

    [Fact]
    public async Task GPUBenchmarkAsync()
    {
        var device = "cuda:0";
        var input = torch.randn(testDimension, testDimension, device: device);
        var model = new SequentialLinear(testDimension, device);

        await BenchmarkAsync(device, input, model);
    }

    [Fact]
    public async Task CPUBenchmarkAsync()
    {
        var device = "cpu";
        var input = torch.randn(testDimension, testDimension, device: device);
        var model = new SequentialLinear(testDimension, device);

        await BenchmarkAsync(device, input, model);
    }

    [Fact]
    public async Task DynamicLoadingBenchmarkAsync()
    {
        long[] gpuMemory = [0];
        foreach (var memory in gpuMemory)
        {
            await DynamicLoadingBenchmark(memory);
        }
    }

    private async Task DynamicLoadingBenchmark(long gpuMemoryInGB)
    {
        using var _ = NewDisposeScope();
        var device = "meta";
        var model = new SequentialLinear(testDimension, device);
        var sizeInBytes = model.GetSizeInBytes();
        var sizeInGigaBytes = 1.0f * sizeInBytes / 1024 / 1024 / 1024;
        output.WriteLine($"SequentialLinear has {sizeInGigaBytes} GB");
        var deviceSizeMap = new Dictionary<string, long>
        {
            ["cuda:0"] = gpuMemoryInGB * 1024 * 1024 * 1024,
            ["cpu"] = 64L * 1024 * 1024 * 1024,
            ["disk"] = 2L * 1024 * 1024 * 1024 * 1024,
        };

        var deviceMap = model.InferDeviceMapForEachLayer(
            devices: ["cuda:0", "cpu", "disk"],
            deviceSizeMapInByte: deviceSizeMap);

        // pretty print the device map
        var json = JsonSerializer.Serialize(deviceMap, new JsonSerializerOptions { WriteIndented = true });
        output.WriteLine(json);
        model = new SequentialLinear(testDimension, "cpu");
        model = model.ToDynamicLoadingModel(deviceMap, "cuda:0");
        var input = torch.randn(testDimension, testDimension, device: "cuda:0");
        await BenchmarkAsync("cuda:0", input, model);
    }

    private async Task BenchmarkAsync(string device, Tensor input, nn.Module<Tensor, Tensor> model)
    {
        using var __ = torch.no_grad();
        model.eval();
        // warm up
        for (var i = 0; i < preHeat; i++)
        {
            using var ___ = NewDisposeScope();
            var _ = model.forward(input);
        }

        // measure
        // 1000 iterations
        var timer = System.Diagnostics.Stopwatch.StartNew();

        for (var i = 0; i < loops; i++)
        {
            using var ___ = NewDisposeScope();
            var _ = model.forward(input);
        }

        timer.Stop();

        output.WriteLine($"SequentialLinear on {device} took {timer.ElapsedMilliseconds * 1.0f / loops} ms on average");
    }


    private class SequentialLinear : nn.Module<Tensor, Tensor>
    {
        private readonly DynamicLoadingModule<PhiLinear, Tensor, Tensor> linear1;
        private readonly PhiLinear linear2;
        //private readonly DynamicLoadingModule<PhiLinear, Tensor, Tensor> linear3;
        //private readonly DynamicLoadingModule<PhiLinear, Tensor, Tensor> linear4;
        //private readonly DynamicLoadingModule<PhiLinear, Tensor, Tensor> linear5;

        private readonly ModuleList<DynamicLoadingModule<PhiLinear, Tensor, Tensor>> linears;
        public SequentialLinear(int features, string? device = null, int numberOfLayers = 64)
            : base(nameof(SequentialLinear))
        {
            this.linear1 = DynamicLoadingModule<PhiLinear, Tensor, Tensor>.CreateFromModel(new PhiLinear(features, features, device: device));
            this.linear2 = new PhiLinear(features, features, device: device);
            //this.linear3 = DynamicLoadingModule<PhiLinear, Tensor, Tensor>.CreateFromModel(new PhiLinear(features, features, device: device));
            //this.linear4 = DynamicLoadingModule<PhiLinear, Tensor, Tensor>.CreateFromModel(new PhiLinear(features, features, device: device));
            //this.linear5 = DynamicLoadingModule<PhiLinear, Tensor, Tensor>.CreateFromModel(new PhiLinear(features, features, device: device));

            this.linears = new ModuleList<DynamicLoadingModule<PhiLinear, Tensor, Tensor>>();

            for (int i = 0; i < numberOfLayers; i++)
            {
                this.linears.Add(DynamicLoadingModule<PhiLinear, Tensor, Tensor>.CreateFromModel(new PhiLinear(features, features, device: device)));
            }

            this.RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            input = linear1.forward(input);
            input = linear2.forward(input);
            foreach (var linear in linears)
            {
                var v0 = linear.forward(input).DetachFromDisposeScope();
                input.add_(v0);
                v0.Dispose();
            }

            return input;
        }
    }
}
