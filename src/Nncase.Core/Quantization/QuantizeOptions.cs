// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Quantization;

/// <summary>
/// Model Quant Mode.
/// </summary>
public enum ModelQuantMode : int
{
    /// <summary>
    /// no quant.
    /// </summary>
    NoQuant,

    /// <summary>
    /// use ptq.
    /// </summary>
    UsePTQ,

    /// <summary>
    /// use qat.
    /// </summary>
    UseQAT,
}

/// <summary>
/// Quant Type.
/// </summary>
public enum QuantType : int
{
    /// <summary>
    /// uint8.
    /// </summary>
    Uint8,

    /// <summary>
    /// int8.
    /// </summary>
    Int8,

    /// <summary>
    /// int16.
    /// </summary>
    Int16,
}

/// <summary>
/// Fine Tune Weights Method.
/// </summary>
public enum FineTuneWeightsMethod : int
{
    /// <summary>
    /// no fine tune weights.
    /// </summary>
    NoFineTuneWeights,

    /// <summary>
    /// use sqaunt.
    /// </summary>
    UseSquant,

    /// <summary>
    /// use adaround.
    /// </summary>
    UseAdaRound,
}

/// <summary>
/// Calibration Method.
/// </summary>
public enum CalibMethod
{
    /// <summary>
    /// no clip.
    /// </summary>
    NoClip,

    /// <summary>
    /// kld.
    /// </summary>
    Kld,
}

/// <summary>
/// How image calibration samples are fitted to the model input size.
/// </summary>
public enum CalibImageResize
{
    /// <summary>
    /// keep the aspect ratio, center the image and pad with gray 114 (Ultralytics YOLO).
    /// </summary>
    Letterbox,

    /// <summary>
    /// keep the aspect ratio, place the image at the top-left and pad right/bottom with gray 114 (YOLOX).
    /// </summary>
    LetterboxTopLeft,

    /// <summary>
    /// resize to the model input size, ignoring the aspect ratio.
    /// </summary>
    Stretch,
}

/// <summary>
/// quantize options.
/// </summary>
public class QuantizeOptions
{
    /// <summary>
    /// Gets or sets calibrationDataset.
    /// </summary>
    public ICalibrationDatasetProvider? CalibrationDataset { get; set; }

    /// <summary>
    /// Gets or sets calibMethod.
    /// </summary>
    public CalibMethod CalibrationMethod { get; set; } = CalibMethod.NoClip;

    /// <summary>
    /// Gets or sets a value indicating whether enable the Auto bind quant method.
    /// </summary>
    public bool BindQuantMethod { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether enable squant to fine tune weights.
    /// </summary>
    public bool UseSquant { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether enable adaround to fine tune weights.
    /// </summary>
    public bool UseAdaRound { get; set; }

    /// <summary>
    /// Gets or sets quant type.
    /// </summary>
    public DataType QuantType { get; set; } = DataTypes.UInt8;

    /// <summary>
    /// Gets or sets weights quant type.
    /// </summary>
    public DataType WQuantType { get; set; } = DataTypes.UInt8;

    /// <summary>
    /// Gets or sets model quant mode.
    /// </summary>
    public ModelQuantMode ModelQuantMode { get; set; } = ModelQuantMode.NoQuant;

    /// <summary>
    /// Gets or sets import quant scheme.
    /// </summary>
    public string QuantScheme { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets import quant info for inner check, this field should not be displayed config.toml.
    /// </summary>
    public string QuantSchemeInnerCheck { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets a value indicating whether strict mode.
    /// </summary>
    public bool QuantSchemeStrictMode { get; set; }

    /// <summary>
    /// Gets or sets the quant scheme the importer derived from the QuantizeLinear/DequantizeLinear pairs of the model.
    /// It is used when no <see cref="QuantScheme"/> file is given and is always strict.
    /// </summary>
    public QuantScheme? ImportedQuantScheme { get; set; }

    /// <summary>
    /// Gets a value indicating whether ranges come from a quant scheme (file or imported) instead of calibration.
    /// </summary>
    public bool HasQuantScheme => QuantScheme != string.Empty || ImportedQuantScheme != null;

    /// <summary>
    /// Gets a value indicating whether the quant scheme in use is strict.
    /// </summary>
    public bool IsQuantSchemeStrict => QuantScheme != string.Empty ? QuantSchemeStrictMode : ImportedQuantScheme != null;

    /// <summary>
    /// Gets or sets a value indicating whether export quant scheme.
    /// </summary>
    public bool ExportQuantScheme { get; set; }

    /// <summary>
    /// Gets or sets quantize scheme path.
    /// </summary>
    public string ExportQuantSchemePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets a value indicating whether export weight range by channel.
    /// </summary>
    public bool ExportWeightRangeByChannel { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether dump quant error.
    /// </summary>
    public bool DumpQuantError { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether dump quant error using symmetric algorithm for signed mode.
    /// </summary>
    public bool DumpQuantErrorSymmetricForSigned { get; set; }

    /// <summary>
    /// Gets or sets how image calibration samples are resized to the model input.
    /// </summary>
    public CalibImageResize CalibImageResize { get; set; } = CalibImageResize.Letterbox;

    /// <summary>
    /// Gets or sets a value indicating whether image calibration samples use BGR channel order instead of RGB.
    /// </summary>
    public bool CalibBGR { get; set; }

    /// <summary>
    /// Creates no quantization options.
    /// </summary>
    /// <returns>No quant options.</returns>
    public static QuantizeOptions CreateNoQuant() => new QuantizeOptions();
}
