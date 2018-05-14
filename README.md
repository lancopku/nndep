# nndep

A transition-based dependency parser with neural networks and hybrid oracle in C#. 

Implementation of the paper _Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese Dependency Parsing_ [[pdf]](https://arxiv.org/pdf/1711.10163) by Xuancheng Ren and Xu Sun. 

# Usage

## Requirements

- Targeting Microsoft .NET Framework 4.6.1+
- Compatible versions of Mono should work fine
- Developed with Microsoft Visual Studio 2017

## Data

- (MUST) Files for training and development in [the CoNLL-U format](http://universaldependencies.org/format.html) 
- (OPTIONAL) Files for testing also in [the CoNLL-U format](http://universaldependencies.org/format.html) 
- (OPTIONAL) File of word embeddings. Format: each line starts with the token, and then are the values (in string) of the embedding of the token.
- (OPTIONAL) Trained model files for testing. TODO.

## Run

Compile the code first, or use the executable provided in releases.

Then
```
nndep.exe <config.json>
```
or
```
mono nnmnist.exe <config.json>
```
where <config.json> is a configuration file. If no configuration file is given, a configuration file template will be written to default.json

Oracle type can be specified in the configuration file. Change the _OraType_ field to _standard_ to use the traditional static oracle, and _hybrid_ to use the hybrid oracle.

The program supports three run modes, *train, dev, and test*. Change the _Mode_ field in the configuration file to set the run mode.
- In train mode, train file and dev file must be given. Test file is optional. The program will train the parser using the settings specified in the configuration file. The parsing results and the parsing model will be saved to files. Please do not specify the _Model_ field. The behavior is not specified.
- In dev mode, a parsing model and a dev file must be given. Test file is optional. The settings specified in the configuration file are not effective. The program will evaluate the parsing model on the file(s). The parsing results will be saved to files.
- In test mode, a parsing model must be given. Dev file and test file are optional. If none is given, the program will just exit. The results are exact the same with the dev mode, except that the running time will be recorded.


There is also a seperate python script in [util](./util/) for getting the related statistics of the datasets, and parsing results. The file is not annotated, but I think the code is rather straight-forward.

# Citation

If you base your research on this paper, please cite [_Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese Dependency Parsing_](https://arxiv.org/pdf/1711.10163) as
```
@article{ren2017hybrid,
  author    = {Xuancheng Ren and
               Xu Sun},
  title     = {Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese
               Dependency Parsing},
  journal   = {CoRR},
  volume    = {abs/1711.10163},
  year      = {2017},
  url       = {http://arxiv.org/abs/1711.10163},
  archivePrefix = {arXiv},
  eprint    = {1711.10163}
}
```
