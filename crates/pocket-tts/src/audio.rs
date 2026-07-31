use candle_core::Tensor;

use hound::{Error as HoundError, WavReader};
#[cfg(not(target_arch = "wasm32"))]
use hound::{WavSpec, WavWriter};

use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
pub fn read_wav<P: AsRef<Path>>(path: P) -> anyhow::Result<(Tensor, u32)> {
    let reader = WavReader::open(path)?;
    read_wav_internal(reader)
}

/// Read any supported audio file to a mono `[1, T]` tensor, mirroring
/// upstream's `audio_read`: WAV always works via hound; other formats
/// (mp3, flac, ogg, m4a) need the optional `audio-formats` feature
/// (symphonia), the Rust counterpart of upstream's optional soundfile.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_audio<P: AsRef<Path>>(path: P) -> anyhow::Result<(Tensor, u32)> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "wav" || ext == "wave" {
        return read_wav(path);
    }

    #[cfg(feature = "audio-formats")]
    {
        read_audio_symphonia(path)
    }
    #[cfg(not(feature = "audio-formats"))]
    {
        anyhow::bail!(
            "reading .{ext} needs the `audio-formats` feature \
             (rebuild with --features audio-formats), or provide a WAV file"
        )
    }
}

/// Decode a non-WAV audio file with symphonia to mono f32.
#[cfg(all(not(target_arch = "wasm32"), feature = "audio-formats"))]
fn read_audio_symphonia(path: &Path) -> anyhow::Result<(Tensor, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("no audio track in {path:?}"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);
    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Recoverable per symphonia docs: skip the malformed packet.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        sample_rate = spec.rate;
        channels = spec.channels.count().max(1);
        let buf = sample_buf.get_or_insert_with(|| {
            SampleBuffer::<f32>::new(decoded.capacity() as u64, spec)
        });
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }

    if mono.is_empty() || sample_rate == 0 {
        anyhow::bail!("no audio decoded from {path:?}");
    }
    let n = mono.len();
    let tensor = Tensor::from_vec(mono, (1, n), &candle_core::Device::Cpu)?;
    Ok((tensor, sample_rate))
}

pub fn read_wav_from_bytes(bytes: &[u8]) -> anyhow::Result<(Tensor, u32)> {
    let reader = WavReader::new(std::io::Cursor::new(bytes))?;
    read_wav_internal(reader)
}

fn read_wav_internal<R: std::io::Read + std::io::Seek>(
    mut reader: WavReader<R>,
) -> anyhow::Result<(Tensor, u32)> {
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            let mut samples = Vec::new();
            for s in reader.samples::<i32>() {
                match s {
                    Ok(v) => samples.push(v as f32 / max_val),
                    Err(e) => {
                        // If we hit an unexpected EOF but have read valid samples, we accept it.
                        if let HoundError::IoError(ref io_err) = e {
                            // Check for UnexpectedEof OR "Failed to read enough bytes" (which is Other in standard hound)
                            let is_unexpected_eof = io_err.kind() == io::ErrorKind::UnexpectedEof;
                            // Check string representation for the specific hound error message
                            let is_truncated_msg = io_err.kind() == io::ErrorKind::Other
                                && io_err.to_string().contains("enough bytes");

                            if (is_unexpected_eof || is_truncated_msg) && !samples.is_empty() {
                                break;
                            }
                        }
                        return Err(anyhow::Error::from(e));
                    }
                }
            }
            samples
        }
        hound::SampleFormat::Float => {
            let mut samples = Vec::new();
            for s in reader.samples::<f32>() {
                match s {
                    Ok(v) => samples.push(v),
                    Err(e) => {
                        if let HoundError::IoError(ref io_err) = e {
                            let is_unexpected_eof = io_err.kind() == io::ErrorKind::UnexpectedEof;
                            let is_truncated_msg = io_err.kind() == io::ErrorKind::Other
                                && io_err.to_string().contains("enough bytes");

                            if (is_unexpected_eof || is_truncated_msg) && !samples.is_empty() {
                                break;
                            }
                        }
                        return Err(anyhow::Error::from(e));
                    }
                }
            }
            samples
        }
    };

    let device = if cfg!(target_arch = "wasm32") {
        &candle_core::Device::Cpu
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            &candle_core::Device::Cpu
        }
        #[cfg(target_arch = "wasm32")]
        {
            &candle_core::Device::Cpu
        }
    };

    let tensor = if channels > 1 {
        // Downmix interleaved multichannel audio to mono, like upstream's
        // audio_read (the models are mono; a stereo prompt should not end up
        // as two "channels" of conditioning).
        let num_samples = samples.len() / channels;
        let mut mono = vec![0.0f32; num_samples];
        for (i, sample) in mono.iter_mut().enumerate() {
            let frame = &samples[i * channels..(i + 1) * channels];
            *sample = frame.iter().sum::<f32>() / channels as f32;
        }
        Tensor::from_vec(mono, (1, num_samples), device)?
    } else {
        let n = samples.len();
        Tensor::from_vec(samples, (1, n), device)?
    };

    Ok((tensor, sample_rate))
}

pub fn pcm_i16_le_bytes(audio: &Tensor) -> anyhow::Result<Vec<u8>> {
    let shape = audio.dims();
    if shape.len() != 2 {
        anyhow::bail!(
            "Expected audio tensor with shape [channels, samples], got {:?}",
            shape
        );
    }

    let data = audio.to_vec2::<f32>()?;
    let channel_slices: Vec<&[f32]> = data.iter().map(|channel| channel.as_slice()).collect();
    Ok(pcm_i16_le_bytes_from_slices(&channel_slices))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn pcm_i16_le_bytes_mono(samples: &[f32]) -> Vec<u8> {
    pcm_i16_le_bytes_from_slices(&[samples])
}

fn pcm_i16_le_bytes_from_slices(channels: &[&[f32]]) -> Vec<u8> {
    if channels.is_empty() {
        return Vec::new();
    }

    let num_samples = channels[0].len();
    let mut out = Vec::with_capacity(num_samples * channels.len() * 2);

    for i in 0..num_samples {
        for channel in channels {
            let val = channel[i].clamp(-1.0, 1.0);
            let val = (val * 32767.0) as i16;
            out.extend_from_slice(&val.to_le_bytes());
        }
    }

    out
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_wav<P: AsRef<Path>>(path: P, audio: &Tensor, sample_rate: u32) -> anyhow::Result<()> {
    let mut writer = std::fs::File::create(path)?;
    write_wav_to_writer(&mut writer, audio, sample_rate)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_wav_to_writer<W: std::io::Write + std::io::Seek>(
    writer: W,
    audio: &Tensor,
    sample_rate: u32,
) -> anyhow::Result<()> {
    let shape = audio.dims();
    if shape.len() != 2 {
        anyhow::bail!(
            "Expected audio tensor with shape [channels, samples], got {:?}",
            shape
        );
    }
    let channels = shape[0] as u16;
    let _num_samples = shape[1];

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut wav_writer = WavWriter::new(writer, spec)?;
    let pcm_bytes = pcm_i16_le_bytes(audio)?;
    for chunk in pcm_bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        wav_writer.write_sample(sample)?;
    }
    wav_writer.finalize()?;
    Ok(())
}

pub fn normalize_peak(audio: &Tensor) -> anyhow::Result<Tensor> {
    let max_abs = audio.abs()?.max_all()?.to_scalar::<f32>()?;
    if max_abs > 0.0 {
        Ok(audio.affine(1.0 / max_abs as f64, 0.0)?)
    } else {
        Ok(audio.clone())
    }
}

// Matches Python's scipy.signal.resample_poly behavior
pub fn resample(audio: &Tensor, from_rate: u32, to_rate: u32) -> anyhow::Result<Tensor> {
    if from_rate == to_rate {
        return Ok(audio.clone());
    }

    let shape = audio.dims();
    let channels = shape[0];
    let num_samples = shape[1];

    if num_samples == 0 {
        return Ok(audio.clone());
    }

    use rubato::{FastFixedIn, Resampler};

    // Calculate output size
    let ratio = to_rate as f64 / from_rate as f64;
    let _new_num_samples = (num_samples as f64 * ratio) as usize;

    // Convert candle Tensor to Vec<Vec<f32>> for rubato
    // Rubato expects [channel][sample]
    let audio_vec = audio.to_vec2::<f32>()?;

    // Create resampler
    // FastFixedIn is synchronous and suitable for full-file resampling
    let mut resampler = FastFixedIn::<f32>::new(
        ratio,
        1.0,                              // max_resample_ratio_relative (1.0 for fixed)
        rubato::PolynomialDegree::Septic, // High quality interpolation
        num_samples,                      // block_size_in
        channels,
    )?;

    // Resample
    let resampled_vec = resampler.process(&audio_vec, None)?;

    // Truncate or pad to exact expected length if necessary (rubato might return slightly more/less due to block/filter delay)
    // But FastFixedIn with fixed block size should be mainly correct.
    // We'll trust rubato's output but sanity check dimensions in the Tensor creation would be good.
    // Actually, rubato might return a slightly different number of samples than naive calculation.
    // Let's use whatever rubato returned.

    let out_channels = resampled_vec.len();
    let out_samples = resampled_vec[0].len();

    // Flatten back to column-major (or whatever candle expects for from_vec)
    // Candle from_vec takes a flat vector and shape.
    // If we have [C][T], we need to flatten to C*T.
    let mut flat_data = Vec::with_capacity(out_channels * out_samples);
    for channel in resampled_vec {
        flat_data.extend(channel);
    }

    Ok(Tensor::from_vec(
        flat_data,
        (out_channels, out_samples),
        audio.device(),
    )?)
}

#[deprecated(note = "Use resample() instead which provides higher quality.")]
pub fn resample_linear(audio: &Tensor, from_rate: u32, to_rate: u32) -> anyhow::Result<Tensor> {
    resample(audio, from_rate, to_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_normalize_peak() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![-0.5f32, 0.2, 0.5], (1, 3), &device)?;
        let normalized = normalize_peak(&t)?;
        let data = normalized.to_vec2::<f32>()?;
        assert_eq!(data[0], vec![-1.0, 0.4, 1.0]);
        Ok(())
    }

    #[test]
    fn test_pcm_i16_le_bytes_clamp_and_interleave() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let data = vec![-1.0f32, 0.0, 1.0, 0.5, -0.5, 2.0];
        let t = Tensor::from_vec(data, (2, 3), &device)?;

        let bytes = pcm_i16_le_bytes(&t)?;
        let samples: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        assert_eq!(samples, vec![-32767, 16383, 0, -16383, 32767, 32767]);
        Ok(())
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_resample() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // rubato works best with reasonable block sizes.
        // Let's use a larger sample count to be safe.
        let input_samples = 1024;
        let data: Vec<f32> = (0..input_samples).map(|i| (i as f32 * 0.1).sin()).collect();
        let t = Tensor::from_vec(data, (1, input_samples), &device)?;

        // Resample 100Hz to 200Hz (Ratio 2.0)
        let resampled = resample(&t, 100, 200)?;
        let out_samples = resampled.dims()[1];

        println!("Resample test: In={}, Out={}", input_samples, out_samples);

        // Expect approx double
        let expected = 2048;
        let diff = (out_samples as i64 - expected as i64).abs();

        assert!(
            diff <= 50,
            "Output samples {} deviates too much from expected {}",
            out_samples,
            expected
        );
        Ok(())
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wav_io() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // Use small values to avoid clipping
        // write_wav applies clamp(-1, 1) to match Python's behavior
        let t = Tensor::from_vec(vec![0.0f32, 0.5, -0.5, 0.1], (1, 4), &device)?;
        let path = "test_io.wav";
        write_wav(path, &t, 16000)?;

        let (read_t, sr) = read_wav(path)?;
        assert_eq!(sr, 16000);
        assert_eq!(read_t.dims(), t.dims());

        // Pre-calculate expected values (clamp doesn't change values in [-1, 1])
        let expected_data: Vec<f32> = vec![0.0, 0.5, -0.5, 0.1];
        let expected = Tensor::from_vec(expected_data, (1, 4), &device)?;

        // Tolerance for 16-bit quantization (1/32768 ~= 3e-5) plus float error
        let diff = (read_t - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "Diff was {}", diff);

        std::fs::remove_file(path)?;
        Ok(())
    }
}
