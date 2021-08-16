# Particle picker modifications

We made the following modifications to particle pickers to allow them to run successfully on our systems.

## ASPIRE-APPLE picker

- Classifier probabilities are not written to output .star files by default
- In helper.py, query block sizes are incorrect if either dimension of the micrograph (in pixels) is divisible by the block size.

## PARSED

- Unintended behavior is observed in autopick_columns.star output files, since Python list values were not comma-separated.
- The x and y coordinates produced in output *.star files are reversed with respect to those columns in corresponding *_param.star files.