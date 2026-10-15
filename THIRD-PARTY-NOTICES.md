# Third-party notices

This project's own code is MIT-licensed (see [LICENSE](LICENSE)). It includes, or is derived from, the third-party material below.

## YAMNet (Apache License 2.0)

- **Files:** source in `model/yamnet/*.py`, the class map `model/yamnet/yamnet_class_map.csv` (also packaged in the `.eap` as `model/yamnet_class_map.csv`), and the pretrained weights embedded in the exported model `traindet.tflite`.
- **Copyright:** 2019 The TensorFlow Authors.
- **Source:** https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- **License:** Apache License 2.0; the full text is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
- **Modifications:** the exported model fixes the input to one 15,600-sample window and uses YAMNet's TFLite-compatible STFT. It also adds a trained classification head (see below) after YAMNet's embedding and class-score outputs.

## Axis ACAP Native SDK examples (Apache License 2.0)

- **What was used:** the audio capture (PipeWire), FastCGI, AXEvent, AXParameter and larod usage in `acap/app/` follows the patterns of the official examples.
- **Copyright:** Axis Communications AB.
- **Source:** https://github.com/AxisCommunications/acap-native-sdk-examples
- **License:** Apache License 2.0; see [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).

## Training data for the bundled train-detection head

The trained head weights embedded in `traindet.tflite` were learned from these datasets:

- **ESC-50** by K. J. Piczak. The dataset is licensed **CC BY-NC 3.0**, and individual clips are CC0, CC-BY or CC-BY-NC. https://github.com/karolpiczak/ESC-50
- **FSD50K** by E. Fonseca et al., Music Technology Group, Universitat Pompeu Fabra. The dataset is CC-BY 4.0, and individual Freesound clips are CC0, CC-BY, CC-BY-NC or Sampling+. https://zenodo.org/records/4060432

Because part of that data is licensed **non-commercial**, the bundled model is provided for **evaluation and non-commercial use only**. The pipeline in `model/` records each clip's license in `model/data/manifest.csv`. For commercial use, retrain on CC0/CC-BY clips (with attribution) and your own recordings.

No dataset audio is included in this repository.

## Runtime libraries (not redistributed)

These libraries are provided by AXIS OS on the device and dynamically linked:

- GLib (LGPL-2.1+)
- PipeWire (MIT)
- Jansson (MIT)
- FastCGI development kit (permissive, Open Market license)
- larod, AXEvent and AXParameter (Axis)

No copies are included in this repository or in the `.eap` package.
