# BioML
Repository to store my BioInformatics related ML solutions

## Model ##

Protein Function CNN Model - A convolutional neural net model for identifying protein function from

Within the last few years the complete sequence has been determined for over 3000
genomes. This has created the need for fully automated methods to analyse the vast amount of
sequence data now available. The assignment of a function for a given protein has proved to be
difficult where no clear homology to proteins of known function exists. Knowing the subcellular
location of a protein (i.e. where in the cell it is found) in may give some clue as to its possible function,
making an automated method that assigns proteins to a certain subcellular location a useful tool for
analysis. In this research I apply Convolutional Neural Networks (CNN) to this task by employing a
fixed sampling strategy to resize variable length sequences into a consistent shape suitable for CNNs.

The hypothesis for applying CNNs to predicting subcellular
classification and not just visualization of genome sequence
is as follows. Images are self-contained sets of data in which
all the information to decode them is present. Furthermore
each pixel has a high correlation to the pixel next to it such
that pixels can be removed without too detrimental an impact
on the image quality (as is done in image compression). It’s
hypothesized that genomic sequences are both complete (no
other information is required by the transcription process to
determine the subcellular location) and exhibit similar
correlation tendencies from one part of the sequence to the
next (which is to say chunks of sequences can be ignored
without too detrimental an effect).

Results:

On the held-out validation set the accuracy was 51%.
Examination at a location level showed the MCC
for Cytosol was particularly low.

It was noticed that accuracy would be high during training
phase but low on the blind test validation suggesting a case of
over-traning despite the use of K-folds cross-validation and
drop-out within layers.
In addition it was found that the inclusion of just one handcrafted
feature, that of sequence length would cause the
model accuracy to remain below 30% during training. It is
hypothesized that providing such a feature would cause the
neural-net to overly depend upon the feature rather than
having “learnt” the feature over a gradual iterative process.

While the results are low the it is not felt that different
parameters of the CNN model were tried out sufficiently to
rule out CNNs as a potential method for subcellular location
classification. Potential areas of investigation would be to
vary with filter window length and adding more layers.
