using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.IR;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Nncase.Quantization;

/// <summary>
/// <see cref="ICalibrationDatasetProvider"/> for standard image dataset folders (JPG, PNG, BMP).
/// Performs letterbox resizing and float32 normalization (0..1) for YOLO models.
/// </summary>
public sealed class ImageCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    private static readonly string[] SupportedExtensions = { ".jpg", ".jpeg", ".png", ".bmp" };

    public ImageCalibrationDatasetProvider(IReadOnlyList<Var> vars, string datasetPath)
    {
        Trace.Assert(Directory.Exists(datasetPath), "The dataset folder path must be valid!");
        Trace.Assert(vars.Count == 1, "Currently ImageProvider supports single-input models (e.g. YOLO).");

        var inputVar = vars[0];
        Trace.Assert(inputVar.CheckedType is TensorType, "Input var must be a TensorType.");

        var tensorType = (TensorType)inputVar.CheckedType;
        Trace.Assert(tensorType.Shape.IsFixed, "Model input shape must be fixed for image dataset provider.");

        // Expected NCHW shape, eg [1, 3, 640, 640] or [1, 3, 320, 320]
        int[] shape = tensorType.Shape.ToValueArray();
        Trace.Assert(shape.Length == 4 && shape[1] == 3, "Expected NCHW tensor layout with 3 channels.");

        int targetWidth = shape[3];
        int targetHeight = shape[2];

        var imageFiles = Directory.EnumerateFiles(datasetPath)
            .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .ToList();

        Trace.Assert(imageFiles.Count > 0, $"No supported images found in directory: {datasetPath}");

        Count = imageFiles.Count;

        Samples = imageFiles.Select(filePath =>
        {
            var values = new Dictionary<Var, IValue>();
            float[] tensorData = ProcessImageToNchwFloat32(filePath, targetWidth, targetHeight);            

            var tensor = Tensor.From<float>(tensorData, shape);
            values.Add(inputVar, Value.FromTensor(tensor));

            return values;
        }).ToAsyncEnumerable();
    }

    public int? Count { get; }

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

    /// <summary>
    /// Loads image, resizes it with Letterbox padding (gray 114), transforms to RGB Float32 [0..1] NCHW layout.
    /// </summary>
    private float[] ProcessImageToNchwFloat32(string imagePath, int targetWidth, int targetHeight)
    {
        using var image = Image.Load<Rgb24>(imagePath);

        // 1. Calculate Letterbox proportions
        float scale = Math.Min((float)targetWidth / image.Width, (float)targetHeight / image.Height);
        int newWidth = (int)Math.Round(image.Width * scale);
        int newHeight = (int)Math.Round(image.Height * scale);

        int padX = (targetWidth - newWidth) / 2;
        int padY = (targetHeight - newHeight) / 2;

        // 2. Resize original image
        image.Mutate(x => x.Resize(newWidth, newHeight));

        // 3. Create target canvas filled with 114 gray (YOLO standard letterbox background)
        var conf = new SixLabors.ImageSharp.Configuration();
        using var canvas = new Image<Rgb24>(conf, targetWidth, targetHeight, new Rgb24(114, 114, 114));
        canvas.Mutate(x => x.DrawImage(image, new Point(padX, padY), 1f));

        // 4. Convert to Float32 NCHW (R-plane, G-plane, B-plane) normalized to 0.0 - 1.0
        float[] nchwBuffer = new float[3 * targetHeight * targetWidth];
        int planeSize = targetHeight * targetWidth;

        canvas.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    int pixelIndex = y * targetWidth + x;
                    nchwBuffer[0 * planeSize + pixelIndex] = pixelRow[x].R / 255.0f; // Red channel
                    nchwBuffer[1 * planeSize + pixelIndex] = pixelRow[x].G / 255.0f; // Green channel
                    nchwBuffer[2 * planeSize + pixelIndex] = pixelRow[x].B / 255.0f; // Blue channel
                }
            }
        });

        return nchwBuffer;
    }
}
