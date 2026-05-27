# CTC from Scratch

This project implements the Connectionist Temporal Classification (CTC)
algorithm from scratch, following:

[A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks, 2006](https://dl.acm.org/doi/10.1145/1143844.1143891) 

## Disclaimer
This repository is a work in progress.

The current focus is on understanding and implementing CTC for individual
sequences. Batch-mode training is not a primary goal at this stage.

## Content
- [ ] mathematical notes on: 
  - CTC forward-backward recursions
  - classical probability-domain scaling and log-domain scaling
  - operator representations of the CTC dynamic program
- [ ] handwritten CTC loss implementations and gradient 
checks against `torch.nn.CTCLoss`
- [ ] a AN4 ASR training pipeline